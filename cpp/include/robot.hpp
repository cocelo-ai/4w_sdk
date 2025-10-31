#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <iomanip>

#include "fx_client.hpp"  // Native FxCli for UDP communication

namespace robot {

// -------- Exceptions --------
struct RobotEStopError : public std::runtime_error { using std::runtime_error::runtime_error; };
struct RobotSetGainsError : public std::runtime_error { using std::runtime_error::runtime_error; };
struct RobotSleepError : public std::runtime_error { using std::runtime_error::runtime_error; };

class Robot {
public:
    Robot()
        : _last_action_len(16),
          _motor_ids_front{1u,2u,3u,4u,5u,6u,7u,8u},
          _motor_ids_rear{9u,10u,11u,12u,13u,14u,15u,16u},
          _cli_disconn_timeout_ms(200),
          _cli_disconn_duration_ms(0),
          _cli_missed_req(0),
          _kp(_last_action_len, 0.0f),
          _kd(_last_action_len, 0.0f),
          _gains_set(false),
          _cli_front("192.168.10.10", 5101),   // [FIX] 세미콜론 → 콤마, 멤버 이니셜라이저로 생성
          _cli_rear("192.168.11.10", 5101)     // [FIX] 세미콜론 → 콤마, 멤버 이니셜라이저로 생성
    {                                          // [FIX] 생성자 본문 시작 누락 보완
        // Observation containers (pre-sized & reused)
        _obs["dof_pos"] = std::vector<float>(12, 0.0f);   // 12개 관절 (바퀴 제외)
        _obs["dof_vel"] = std::vector<float>(16, 0.0f);   // 16개 모터 속도 (바퀴 포함)
        _obs["ang_vel"] = std::vector<float>(3, 0.0f);
        _obs["proj_grav"] = std::vector<float>(3, 0.0f);
        _obs["last_action"] = std::vector<float>(16, 0.0f);
        _obs["lin_vel"] = std::vector<float>(3, 0.0f);
        _obs["height_map"] = std::vector<float>(144, 0.6128f);

        // Offsets & limits
        _pos_offset = {
            {"left_hip_f", 0.0f}, {"right_hip_f", 0.0f},
            {"left_shoulder_f", 0.0f}, {"right_shoulder_f", 0.0f},
            {"left_leg_f", 0.0f}, {"right_leg_f", 0.0f},

            {"left_hip_r", 0.0f}, {"right_hip_r", 0.0f},
            {"left_shoulder_r", 0.0f}, {"right_shoulder_r", 0.0f},
            {"left_leg_r", 0.0f}, {"right_leg_r", 0.0f},
        };
        _rel_max_pos = {
            {"left_hip_f", 3.14f}, {"right_hip_f", 3.14f},
            {"left_shoulder_f", 3.14f}, {"right_shoulder_f", 3.14f},
            {"left_leg_f", 3.14f}, {"right_leg_f", 3.14f},

            {"left_hip_r", 3.14f}, {"right_hip_r", 3.14f},
            {"left_shoulder_r", 3.14f}, {"right_shoulder_r", 3.14f},
            {"left_leg_r", 3.14f}, {"right_leg_r", 3.14f},
        };
        _rel_min_pos = {
            {"left_hip_f", -3.14f}, {"right_hip_f", -3.14f},
            {"left_shoulder_f", -3.14f}, {"right_shoulder_f", -3.14f},
            {"left_leg_f", -3.14f}, {"right_leg_f", -3.14f},

            {"left_hip_r", -3.14f}, {"right_hip_r", -3.14f},
            {"left_shoulder_r", -3.14f}, {"right_shoulder_r", -3.14f},
            {"left_leg_r", -3.14f}, {"right_leg_r", -3.14f},
        };
        _joint_names = { // pos 인덱스 0..11에 해당하는 관절 이름
            "left_hip_f","right_hip_f","left_shoulder_f","right_shoulder_f","left_leg_f","right_leg_f",
            "left_hip_r","right_hip_r","left_shoulder_r","right_shoulder_r","left_leg_r","right_leg_r"
        };

        _wait(); // [FIX] 양쪽 보드 준비 대기
    }

    // ------- Set gains -------
    void set_gains(const std::vector<float>& kp, const std::vector<float>& kd) {
        if (kp.size() != _last_action_len)
            throw RobotSetGainsError("kp length mismatch for the robot.");
        if (kd.size() != _last_action_len)
            throw RobotSetGainsError("kd length mismatch for the robot.");
        if (kp[6] != 0.0f || kp[7] != 0.0f)
            throw RobotSetGainsError("Wheel motor kp must be zero for indices 6 and 7.");
        if (kp[14] != 0.0f || kp[15] != 0.0f)
            throw RobotSetGainsError("Wheel motor kp must be zero for indices 14 and 15.");
        for (float v : kp) if (v < 0.0f) throw RobotSetGainsError("kp must be non-negative.");
        for (float v : kd) if (v < 0.0f) throw RobotSetGainsError("kd must be non-negative.");
        _kp = kp; _kd = kd; _gains_set = true;
    }

    // ------- Safety check -------
    void check_safety() { // [FIX] 잘못된 시그니처(void check_safety(name={...})) 수정, 양쪽 보드 모두 점검
        std::string status_front = _cli_front.status();
        std::string status_rear  = _cli_rear.status();

        auto [dis_f, emg_f] = _check_status(status_front, _motor_ids_front); // [FIX] 보드별 상태 점검
        auto [dis_r, emg_r] = _check_status(status_rear,  _motor_ids_rear);

        bool disconn_flag = dis_f || dis_r;
        bool emergency_flag = emg_f || emg_r;

        if (!disconn_flag) _cli_disconn_duration_ms = 0;
        else _cli_disconn_duration_ms += 20;

        if (emergency_flag || std::max(_cli_disconn_duration_ms, _cli_missed_req * 20) >= _cli_disconn_timeout_ms)
            throw RobotEStopError("E-stop: connection timeout or emergency flag reported");

        _check_obs(_obs);
    }

    // ------- Observation (returns copy; internal buffers reused) -------
    std::unordered_map<std::string, std::vector<float>> get_obs() { // [FIX] 전/후 보드 모두에서 수집
        std::string mcu_front = _cli_front.req(_motor_ids_front);
        std::string mcu_rear  = _cli_rear.req(_motor_ids_rear);
        auto& parsed = _parse_obs(mcu_front, mcu_rear);  // [FIX] 두 문자열을 한 번에 파싱
        return parsed;
    }

    // ------- Action -------
    void do_action(const std::vector<float>& action, bool torque_ctrl=false) {
        if (!_gains_set)
            throw RobotSetGainsError("Robot's kp and kd must be provided before do_action.");
        if (action.size() != _last_action_len)
            estop("action length mismatch.");

        std::vector<float> pos(_last_action_len, 0.0f);
        std::vector<float> vel(_last_action_len, 0.0f);
        std::vector<float> kp(_last_action_len, 0.0f);
        std::vector<float> kd(_last_action_len, 0.0f);
        std::vector<float> tau(_last_action_len, 0.0f);

        if (torque_ctrl) {
            tau = action;
        } else {
            // [FIX] 16채널 매핑:
            //  - 0..5  : 앞다리 6관절 위치 제어
            //  - 6..7  : 앞바퀴 속도 제어
            //  - 8..13 : 뒷다리 6관절 위치 제어
            //  - 14..15: 뒷바퀴 속도 제어
            for (size_t i = 0; i < _last_action_len; ++i) {
                bool is_pos_idx = (i < 6) || (i >= 8 && i < 14);
                if (is_pos_idx) {
                    // pos 인덱스 → _joint_names 인덱스 매핑
                    size_t jidx = (i < 6) ? i : (i - 2); // [FIX] 8..13 -> 6..11
                    const std::string& jname = _joint_names[jidx];
                    float off = _pos_offset[jname];
                    pos[i] = action[i] - off;
                } else {
                    vel[i] = action[i]; // 바퀴(6,7,14,15)는 속도 제어
                }
                kp[i] = _kp[i];
                kd[i] = _kd[i];
            }
        }

        // [FIX] 전/후 보드로 분리 송신
        auto slice = [](const std::vector<float>& v, size_t s, size_t e) {
            return std::vector<float>(v.begin()+s, v.begin()+e);
        };
        // 앞 보드: 인덱스 0..7
        _cli_front.operation_control(
            _motor_ids_front,
            slice(pos, 0, 8), slice(vel, 0, 8),
            slice(kp,  0, 8), slice(kd,  0, 8),
            slice(tau, 0, 8)
        );
        // 뒤 보드: 인덱스 8..15
        _cli_rear.operation_control(
            _motor_ids_rear,
            slice(pos, 8, 16), slice(vel, 8, 16),
            slice(kp,  8, 16), slice(kd,  8, 16),
            slice(tau, 8, 16)
        );
        // (선택) last_action 저장
        _obs["last_action"] = action;
        check_safety();
    }

    // ------- Control utils -------
    [[noreturn]] void estop(const std::string& msg = std::string()) {
        const auto retry = std::chrono::milliseconds(10);
        for (;;) {
            bool ok_f = _cli_front.motor_estop(_motor_ids_front); // [FIX] 양쪽 보드 E-stop
            bool ok_r = _cli_rear.motor_estop(_motor_ids_rear);   // [FIX]
            if (ok_f && ok_r) break;
            std::this_thread::sleep_for(retry);
        }
        throw RobotEStopError(msg.empty() ? "E-stop triggered" : msg);
    }

    [[noreturn]] void sleep() {
        const auto retry = std::chrono::milliseconds(10);
        for (;;) {
            bool ok_f = _cli_front.motor_estop(_motor_ids_front); // [FIX] 양쪽 보드 Sleep=E-stop
            bool ok_r = _cli_rear.motor_estop(_motor_ids_rear);   // [FIX]
            if (ok_f && ok_r) break;
            std::this_thread::sleep_for(retry);
        }
        throw RobotSleepError("Sleep triggered");
    }

    void stand() { /* TODO */ }
    void precise_stop() { /* TODO */ }

private:
    // HW 준비 대기
    void _wait(std::int32_t timeout_ms = 30000) { // [FIX] 양쪽 보드 모두 준비될 때까지 대기
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        const auto retry_sleep = std::chrono::milliseconds(100);
        const auto safe_margin = std::chrono::milliseconds(100);

        while (std::chrono::steady_clock::now() < deadline) {
            bool started_f = _cli_front.motor_start(_motor_ids_front);
            bool started_r = _cli_rear.motor_start(_motor_ids_rear);
            if (!(started_f && started_r)) {
                std::this_thread::sleep_for(retry_sleep);
                continue;
            }

            std::string status_front = _cli_front.status();
            std::string status_rear  = _cli_rear.status();
            auto [dis_f, emg_f] = _check_status(status_front, _motor_ids_front);
            auto [dis_r, emg_r] = _check_status(status_rear,  _motor_ids_rear);
            if (dis_f || dis_r || emg_f || emg_r) {
                std::this_thread::sleep_for(retry_sleep);
                continue;
            }

            std::this_thread::sleep_for(safe_margin);
            return;
        }
        throw RobotEStopError("Motor start timeout");
    }

    // Status check (native string parsing)
    std::pair<bool,bool> _check_status(const std::string& status_str,
                                       const std::vector<uint8_t>& ids) { // [FIX] 보드별 아이디 집합을 받도록 변경
        bool disconn_flag = false, emergency_flag = false;

        if (status_str.find("OK <STATUS>") == std::string::npos) {
            disconn_flag = true;
        } else {
            for (auto id : ids) {
                std::string mid = "M" + std::to_string(id);
                size_t pos = status_str.find(mid);
                if (pos == std::string::npos) { disconn_flag = true; break; }
                pos = status_str.find("pattern:", pos);
                if (pos == std::string::npos) { disconn_flag = true; break; }
                int pattern = std::stoi(status_str.substr(pos + 8));
                if (pattern != 2) { disconn_flag = true; break; }
            }
        }

        size_t emg_pos = status_str.find("EMERGENCY");
        if (emg_pos != std::string::npos) {
            size_t val_pos = status_str.find("value:", emg_pos);
            if (val_pos != std::string::npos && status_str.substr(val_pos + 6, 2) == "on") {
                emergency_flag = true;
            }
        }
        return {disconn_flag, emergency_flag};
    }

    // MCU data sanity (native string)
    bool _check_mcu_data(const std::string& mcu_str, const std::vector<uint8_t>& ids) {

        if (mcu_str.find("OK <REQ>") == std::string::npos) {
            _cli_missed_req += 1;
            return false;
        }

        // 1.._last_action_len 까지 쓸 거라 +1
        std::vector<std::size_t> motor_pos(_last_action_len + 1, 0);

        // 한 번만 스캔해서 위치 캐싱
        {
            std::size_t cur = 0;
            const std::size_t n = mcu_str.size();
            while (cur < n) {
                std::size_t mpos = mcu_str.find('M', cur);
                if (mpos == std::string::npos) break;

                std::size_t p = mpos + 1;
                int num = 0;
                bool has_digit = false;
                while (p < n && mcu_str[p] >= '0' && mcu_str[p] <= '9') {
                    has_digit = true;
                    num = num * 10 + (mcu_str[p] - '0');
                    ++p;
                }

                // 기대하는 범위 안에 있을 때만 저장
                if (has_digit && num >= 1 && num <= _last_action_len) {
                    motor_pos[num] = mpos;
                }

                cur = p;
            }
        }

        // p, v, t 세 개만 도는 테이블
        static constexpr const char* kKeys[] = { "p:", "v:", "t:" };

        // 필요한 id만 검사
        for (uint8_t id : ids) {
            // 기대한 id가 근처에 아예 없으면 실패
            if (id == 0 || id > _last_action_len) {
                _cli_missed_req += 1;
                return false;
            }

            std::size_t mid_pos = motor_pos[id];
            if (mid_pos == 0) {
                _cli_missed_req += 1;
                return false;
            }

            // 여기만 inner loop로 정리
            for (const char* key : kKeys) {
                std::size_t pos = mcu_str.find(key, mid_pos);
                if (pos == std::string::npos) {
                    _cli_missed_req += 1;
                    return false;
                }
                // "p:" / "v:" / "t:" 다 2글자라 +2 고정
                std::size_t val_pos = pos + 2;
                if (val_pos < mcu_str.size() && mcu_str[val_pos] == 'N') {
                    _cli_missed_req += 1;
                    return false;
                }
            }
        }

        return true;
    }

    // Parse obs (in-place into pre-sized vectors, native string parsing)
    std::unordered_map<std::string, std::vector<float>>&
    _parse_obs(const std::string& mcu_front, const std::string& mcu_rear) {
        // 1) 패킷 유효성 먼저
        if (!_check_mcu_data(mcu_front, _motor_ids_front)) {
            return _obs;
        }
        if (!_check_mcu_data(mcu_rear, _motor_ids_rear)) {
            return _obs;
        }

        // 2) 이 시점에서 "이번 패킷" 안에서는 M 들의 위치가 항상 같다고 가정하니까
        //    멤버 배열이 비어 있으면 그때 한 번만 채운다.
        //    (index 0은 안 쓰니까 1이나 9가 0인지로 판별)
        if (_front_motor_pos[1] == 0) {
            std::size_t cur = 0;
            const std::size_t n = mcu_front.size();
            while (cur < n) {
                std::size_t mpos = mcu_front.find('M', cur);
                if (mpos == std::string::npos) break;

                std::size_t p = mpos + 1;
                int num = 0;
                bool has_digit = false;
                while (p < n && mcu_front[p] >= '0' && mcu_front[p] <= '9') {
                    has_digit = true;
                    num = num * 10 + (mcu_front[p] - '0');
                    ++p;
                }
                if (has_digit && num >= 1 && num <= 8) {
                    _front_motor_pos[num] = mpos;
                }
                cur = p;
            }
        }
        if (_rear_motor_pos[9] == 0) {
            std::size_t cur = 0;
            const std::size_t n = mcu_rear.size();
            while (cur < n) {
                std::size_t mpos = mcu_rear.find('M', cur);
                if (mpos == std::string::npos) break;

                std::size_t p = mpos + 1;
                int num = 0;
                bool has_digit = false;
                while (p < n && mcu_rear[p] >= '0' && mcu_rear[p] <= '9') {
                    has_digit = true;
                    num = num * 10 + (mcu_rear[p] - '0');
                    ++p;
                }
                if (has_digit && num >= 9 && num <= 16) {
                    _rear_motor_pos[num] = mpos;
                }
                cur = p;
            }
        }

        // 3) 이제부터는 멤버에 들어있는 위치만 써서 값 뽑는다
        auto& dof_pos   = _obs["dof_pos"]; // 12
        auto& dof_vel   = _obs["dof_vel"]; // 16
        auto& ang_vel   = _obs["ang_vel"];
        auto& proj_grav = _obs["proj_grav"];

        // ---- 위치 ----
        // front: M1..M6 -> dof_pos[0..5]
        for (int i = 0; i < 6; ++i) {
            int mid = i + 1; // 1..6
            std::size_t base = _front_motor_pos[mid];
            // 여기서는 위치가 있다고 가정했으니까 바로 find
            std::size_t ppos = mcu_front.find("p:", base);
            float val = std::stof(mcu_front.substr(ppos + 2));
            dof_pos[i] = val + _pos_offset[_joint_names[i]];
        }

        // rear: M9..M14 -> dof_pos[6..11]
        for (int j = 0; j < 6; ++j) {
            int mid = 9 + j; // 9..14
            std::size_t base = _rear_motor_pos[mid];
            std::size_t ppos = mcu_rear.find("p:", base);
            float val = std::stof(mcu_rear.substr(ppos + 2));
            dof_pos[6 + j] = val + _pos_offset[_joint_names[6 + j]];
        }

        // ---- 속도 ----
        // front: M1..M8 -> dof_vel[0..7]
        for (int i = 0; i < 8; ++i) {
            int mid = i + 1; // 1..8
            std::size_t base = _front_motor_pos[mid];
            std::size_t vpos = mcu_front.find("v:", base);
            float val = std::stof(mcu_front.substr(vpos + 2));
            dof_vel[i] = val;
        }

        // rear: M9..M16 -> dof_vel[8..15]
        for (int j = 0; j < 8; ++j) {
            int mid = 9 + j; // 9..16
            std::size_t base = _rear_motor_pos[mid];
            std::size_t vpos = mcu_rear.find("v:", base);
            float val = std::stof(mcu_rear.substr(vpos + 2));
            dof_vel[8 + j] = val;
        }

        // ---- IMU (rear) ----
        {
            std::size_t imu_pos = mcu_rear.rfind("IMU");
            if (imu_pos != std::string::npos) {
                std::size_t gx_pos = mcu_rear.find("gx:", imu_pos);
                if (gx_pos != std::string::npos) ang_vel[0] = std::stof(mcu_rear.substr(gx_pos + 3));
                std::size_t gy_pos = mcu_rear.find("gy:", imu_pos);
                if (gy_pos != std::string::npos) ang_vel[1] = std::stof(mcu_rear.substr(gy_pos + 3));
                std::size_t gz_pos = mcu_rear.find("gz:", imu_pos);
                if (gz_pos != std::string::npos) ang_vel[2] = std::stof(mcu_rear.substr(gz_pos + 3));

                std::size_t pgx_pos = mcu_rear.find("pgx:", imu_pos);
                if (pgx_pos != std::string::npos) proj_grav[0] = std::stof(mcu_rear.substr(pgx_pos + 4));
                std::size_t pgy_pos = mcu_rear.find("pgy:", imu_pos);
                if (pgy_pos != std::string::npos) proj_grav[1] = std::stof(mcu_rear.substr(pgy_pos + 4));
                std::size_t pgz_pos = mcu_rear.find("pgz:", imu_pos);
                if (pgz_pos != std::string::npos) proj_grav[2] = std::stof(mcu_rear.substr(pgz_pos + 4));
            }
        }

        return _obs;
    }

    // ------- Obs safety -------
    void _check_obs(const std::unordered_map<std::string, std::vector<float>>& obs) const {
        const auto& q_obs = obs.at("dof_pos"); // 12
        const auto& q_vel = obs.at("dof_vel"); // 16

        const float pos_margin = 0.1745f; // 10 deg
        const float vel_margin = 0.3491f; // 20 deg
        const float vel_th = 8.7275f; // rad/s

        for (size_t i=0;i<_joint_names.size();++i) {
            const std::string& name = _joint_names[i];
            float pos = q_obs.at(i);

            // [FIX] 속도 인덱스 매핑 (앞 0..5 -> 0..5, 뒤 6..11 -> 8..13)
            size_t v_idx = (i < 6) ? i : (i + 2);
            float vel = q_vel.at(v_idx);

            float lo_pos = _rel_min_pos.at(name) + pos_margin;
            float hi_pos = _rel_max_pos.at(name) - pos_margin;

            if (pos < lo_pos || pos > hi_pos) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "E-stop: position limit exceeded on %s (pos=%.3f rad, allowed [%.3f, %.3f])",
                    name.c_str(), pos, lo_pos, hi_pos);
                throw RobotEStopError(buf);
            }
            if (pos < lo_pos + vel_margin && vel < -vel_th) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "E-stop: excessive negative velocity near lower limit on %s (pos=%.3f rad, vel=%.3f rad/s)",
                    name.c_str(), pos, vel);
                throw RobotEStopError(buf);
            }
            if (pos >= hi_pos - vel_margin && vel > vel_th) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "E-stop: excessive positive velocity near upper limit on %s (pos=%.3f rad, vel=%.3f rad/s)",
                    name.c_str(), pos, vel);
                throw RobotEStopError(buf);
            }
        }
    }

private:
    // config / ids
    const size_t _last_action_len;
    std::vector<uint8_t> _motor_ids_front;
    std::vector<uint8_t> _motor_ids_rear;

    // conn state
    int _cli_disconn_timeout_ms;
    int _cli_disconn_duration_ms;
    int _cli_missed_req;

    // state (pre-sized & reused)
    std::unordered_map<std::string, std::vector<float>> _obs;
    std::unordered_map<std::string, float> _pos_offset;
    std::unordered_map<std::string, float> _rel_max_pos, _rel_min_pos;
    std::vector<std::string> _joint_names;

    // gains
    std::vector<float> _kp;
    std::vector<float> _kd;
    bool _gains_set;

    // Native FxCli handle
    FxCli _cli_front;
    FxCli _cli_rear;
};

} // namespace robot
