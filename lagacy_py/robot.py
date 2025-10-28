from typing import List, Dict
import time

import fx_cli
from w4_sdk.core.exceptions import *


class Robot:
    def __init__(self):
        self._cli = fx_cli.FxCli()
        self._obs_to_length = {
            "dof_pos": 6,
            "dof_vel": 8,
            "lin_vel": 3,
            "ang_vel": 3,
            "proj_grav": 3,
            "last_action": 8,
            "height_map": 144
        }
        self._last_action_len = self._obs_to_length["last_action"]
        self._motor_ids = list(range(1, self._last_action_len + 1))
        self._latest_mcu_data = None

        self._cli_disconn_timout_ms = 200
        self._cli_disconn_duration_ms = 0
        self._cli_missed_req = 0

        self._kp = None
        self._kd = None

        self._obs = {
            "dof_pos": [0.0] * 6,
            "dof_vel": [0.0] * 8,
            "ang_vel": [0.0] * 3,
            "proj_grav": [0.0] * 3,
            "last_action": [0.0] * 8,
            "height_map": [0.6128] * 144,
        }
        self._pos_offset = {
            "left_hip": 0.0,
            "right_hip": 0.0,
            "left_shoulder": 0.0,
            "right_shoulder": 0.0,
            "left_leg": 0.0,
            "right_leg": 0.0,
        }

        self._rel_max_pos = {
            "left_hip": 3.14,
            "right_hip": 3.14,
            "left_shoulder": 3.14,
            "right_shoulder": 3.140,
            "left_leg": 3.14,
            "right_leg": 3.14,
        }

        self._rel_min_pos = {
            "left_hip": -3.14,
            "right_hip": -3.14,
            "left_shoulder": -3.14,
            "right_shoulder": -3.14,
            "left_leg": -3.14,
            "right_leg": -3.14,
        }

        self._joint_names = ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_leg", "right_leg"]
        self._mids = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']

        self._wait()

    def _wait(self, timeout_ms: int = 30_000):
        retry_sleep_ms = 100
        safe_margin_ms = 100

        deadline = time.monotonic() + timeout_ms / 1000.0

        while time.monotonic() < deadline:
            if not self._cli.motor_start(self._motor_ids):
                time.sleep(min(retry_sleep_ms / 1000, max(0.0, deadline - time.monotonic())))
                continue

            status = self._cli.status()
            disconn_flag, emergency_flag = self._check_status(status)
            if disconn_flag or emergency_flag:
                time.sleep(min(retry_sleep_ms / 1000, max(0.0, deadline - time.monotonic())))
                continue

            time.sleep(min(safe_margin_ms / 1000, max(0.0, deadline - time.monotonic())))
            return

        raise RobotEStopError(f"Motor start timeout after {timeout_ms} ms")

    def _check_status(self, status):
        disconn_flag = False
        emergency_flag = False

        if status.get("OK <STATUS>") is None:
            disconn_flag = True
        else:
            for mid in self._mids:
                if status[mid]["pattern"] != 2:
                    disconn_flag = True

        emergency_filed = status.get("EMERGENCY")
        if emergency_filed is not None:
            if emergency_filed.get("value") == "on":
                emergency_flag = True

        return disconn_flag, emergency_flag

    def _check_mcu_data(self, mcu_data) -> bool:
        if mcu_data.get("OK <REQ>") is None:
            self._cli_missed_req += 1
            return False

        for mid in self._mids:
            for key in ['p', 'v', 't']:
                if 'N' == mcu_data[mid][key]:
                    self._cli_missed_req += 1
                    return False

        self._cli_missed_req = 0
        return True

    def set_gains(self, kp: list[float], kd: list[float]):
        if not isinstance(kp, list) or not isinstance(kd, list):
            raise RobotSetGainsError("kp and kd Gains must be lists or tuples.")

        if len(kp) != self._last_action_len:
            raise RobotSetGainsError("kp Gains length mismatch for the robot.")
        if len(kd) != self._last_action_len:
            raise RobotSetGainsError("kd Gains length mismatch for the robot.")

        for i in [6, 7]:
            if kp[i] != 0:
                raise RobotSetGainsError(f"Wheel motor kp must be zero, bug got ({kp[6], kp[7]})")

        for k in [*kp, *kd]:
            if k < 0:
                raise RobotSetGainsError("kp Gains must be non-negative.")

        self._kp = kp
        self._kd = kd

        try:
            for i in range(self._last_action_len):
                self._kp[i] = float(self._kp[i])
                self._kd[i] = float(self._kd[i])
        except:
            raise RobotSetGainsError

    def _check_obs(self, obs):
        # pos, vel limits check
        q_obs = obs["dof_pos"]
        q_vel = obs["dof_vel"]
        pos_margin = 0.1745  # rad (10 degrees)
        vel_margin = 0.3491  # rad (20 degrees)
        vel_th = 8.7275  # rad/s

        for i, name in enumerate(self._joint_names):
            pos = q_obs[i]
            vel = q_vel[i]

            lo_pos = self._rel_min_pos[name] + pos_margin
            hi_pos = self._rel_max_pos[name] - pos_margin

            if pos < lo_pos or pos > hi_pos:
                raise RobotEStopError(
                    f"E-stop: position limit exceeded on {name} (pos={pos:.3f} rad, allowed [{lo_pos:.3f}, {hi_pos:.3f}])")

            if pos < lo_pos + vel_margin and vel < -vel_th:
                raise RobotEStopError(
                    f"E-stop: excessive negative velocity near lower limit on {name} (pos={pos:.3f} rad, vel={vel:.3f} rad/s)")

            if pos >= hi_pos - vel_margin and vel > vel_th:
                raise RobotEStopError(
                    f"E-stop: excessive positive velocity near lower limit on {name} (pos={pos:.3f} rad, vel={vel:.3f} rad/s)")
        return

    def check_safety(self):
        # status check
        status = self._cli.status()
        disconn_flag, emergency_flag = self._check_status(status)
        if not disconn_flag:
            self._cli_disconn_duration_ms = 0
        else:
            self._cli_disconn_duration_ms += 20

        if emergency_flag or max(self._cli_disconn_duration_ms,
                                 self._cli_missed_req * 20) >= self._cli_disconn_timout_ms:
            raise RobotEStopError("E-stop: connection timeout or emergency flag reported")

        obs = self.get_obs()
        self._check_obs(obs)
        return

    def _parse_obs(self, mcu_data) -> Dict[str, List]:
        if self._check_mcu_data(mcu_data) is False:
            return self._obs

        self._obs = {"dof_pos": [mcu_data["M1"]["p"] + self._pos_offset["left_hip"],
                                 mcu_data["M2"]["p"] + self._pos_offset["right_hip"],
                                 mcu_data["M3"]["p"] + self._pos_offset["left_shoulder"],
                                 mcu_data["M4"]["p"] + self._pos_offset["right_shoulder"],
                                 mcu_data["M5"]["p"] + self._pos_offset["left_leg"],
                                 mcu_data["M6"]["p"] + self._pos_offset["right_leg"]],
                     "dof_vel": [mcu_data["M1"]["v"], mcu_data["M2"]["v"], mcu_data["M3"]["v"], mcu_data["M4"]["v"],
                                 mcu_data["M5"]["v"], mcu_data["M6"]["v"], mcu_data["M7"]["v"], mcu_data["M8"]["v"]],
                     "ang_vel": [mcu_data["IMU"]["gx"], mcu_data["IMU"]["gy"], mcu_data["IMU"]["gz"]],
                     "proj_grav": [mcu_data["IMU"]["pgx"], mcu_data["IMU"]["pgy"], mcu_data["IMU"]["pgz"]],
                     }
        return self._obs

    def get_obs(self) -> Dict[str, List]:
        mcu_data = self._cli.req(self._motor_ids)
        obs = self._parse_obs(mcu_data)  # TODO: Add height_map
        self._check_obs(obs)
        return obs

    def do_action(self, action: List[float], torque_ctrl: bool = False):
        if self._kp is None or self._kd is None:
            raise RobotSetGainsError("Robot's kp and kd must be provided.")

        # if action이 1D list가 아니면 (첫번째 원소 길이를 열어보는걸로 간단히 2d인지 아닌지 체크 )
        #    self.estop("action must be a 1D list")
        if len(action) != self._last_action_len:
            self.estop(f"action length must be {self._last_action_len}, but got {len(action)}")

        if torque_ctrl:
            motor_cmd = [{"id": mid, "pos": 0.0, "vel": 0.0, "kp": 0.0, "kd": 0.0, "tau": action[mid - 1]} for mid in
                         self._motor_ids]
        else:
            offset = [
                self._pos_offset["left_hip"],  # M1
                self._pos_offset["right_hip"],  # M2
                self._pos_offset["left_shoulder"],  # M3
                self._pos_offset["right_shoulder"],  # M4
                self._pos_offset["left_leg"],  # M5
                self._pos_offset["right_leg"],  # M6
            ]
            motor_cmd = []
            for mid in self._motor_ids:
                i = mid - 1
                if i < 6:
                    pos_sp = action[i] - offset[i]
                    vel_sp = 0.0
                else:
                    pos_sp = 0.0
                    vel_sp = action[i]

                motor_cmd.append({
                    "id": mid,
                    "pos": pos_sp,
                    "vel": vel_sp,
                    "kp": self._kp[i],
                    "kd": self._kd[i],
                    "tau": 0.0
                })

        self._cli.operation_control(motor_cmd)

    def estop(self, e=""):
        retry_sleep_ms = 10
        while True:
            succeed = self._cli.motor_estop(self._motor_ids)
            if succeed:
                break
            time.sleep(retry_sleep_ms / 1000)
        raise RobotEStopError(e)

    def sleep(self):
        # TODO: Change (currently same to estop logic)
        retry_sleep_ms = 10
        while True:
            succeed = self._cli.motor_estop(self._motor_ids)
            if succeed:
                break
            time.sleep(retry_sleep_ms / 1000)
        raise RobotSleepError

    def stand(self):
        # TODO: implement
        pass

    def precise_stop(self):
        # TODO: implement
        pass


