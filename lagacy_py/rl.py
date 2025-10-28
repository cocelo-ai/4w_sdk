from typing import List, Dict, Any, Optional

import numpy as np
from w4_sdk.classes.mode import Mode


class RL:
    def __init__(self):
        self._obs_to_length = {
            "dof_pos": 6,
            "dof_vel": 8,
            "lin_vel": 3,
            "ang_vel": 3,
            "proj_grav": 3,
            "last_action": 8,
            "height_map": 144
        }
        self._mode: Optional[Mode] = None
        self._modes: List[Mode] = []
        self._single_frame = None
        self._single_frame_len = None
        self._state = None
        self._last_action = np.zeros(self._obs_to_length["last_action"], dtype=np.float32)    # action in [-1, 1] (normalized)
        self._scaled_action = np.zeros(self._obs_to_length["last_action"], dtype=np.float32)  # action in [ X, Y] (scaled)

    def _ensure_mode(self):
        if self._mode is None:
            raise RuntimeError("Mode is not set. Call set_mode() first.")

    def add_mode(self, mode: Mode) -> None:
        for idx, m in enumerate(self._modes):
            if m.id == mode.id:
                self._modes[idx] = mode
                if self._mode is not None and self._mode.id == mode.id:
                    self._mode = mode
                return
        self._modes.append(mode)

    def set_mode(self,mode_id: Optional[int]):
        if mode_id is None:
            return

        for mode in self._modes:
            if mode.id == mode_id:
                self._obs_to_length["command"] = mode.cmd_vector_length
                state_len = 0
                for obs in mode.stacked_obs_order:
                    state_len += self._obs_to_length[obs]
                self._single_frame_len = state_len
                self._single_frame = np.zeros(self._single_frame_len, dtype=np.float32)
                state_len *= mode.stack_size

                for obs in mode.non_stacked_obs_order:
                    state_len += self._obs_to_length[obs]

                self._state = np.zeros(state_len, dtype=np.float32)
                self._last_action = np.zeros(self._obs_to_length["last_action"], dtype=np.float32)
                self._mode = mode
                return

    def build_state(self, obs: Dict[str, List], cmd: Dict[str, Any], scaled_last_action=None) -> list:
        if cmd["mode_id"] is not None:
            self.set_mode(mode_id=cmd.get("mode_id"))
        self._ensure_mode()

        if scaled_last_action is not None:
            if len(scaled_last_action) != len(self._last_action):
                raise ValueError(f"unscaled_last_action must have length {len(self._last_action)}")
            self._last_action = np.asarray(scaled_last_action, dtype=np.float32)

        # 1. Build single frame
        i = 0
        for obs_key in self._mode.stacked_obs_order:
            obs_len = self._obs_to_length[obs_key]
            if obs_key == "command":
                obs_vector = cmd.get("cmd_vector")
                scale = self._mode.cmd_scale
            elif obs_key == "last_action":
                obs_vector = self._last_action
                scale = self._mode.obs_scale.get(obs_key, [1.0] * obs_len)
            else:
                obs_vector = obs.get(obs_key)
                scale = self._mode.obs_scale.get(obs_key, [1.0] * obs_len)

            if obs_vector is None:  # If missing, keep old values
                for _ in range(obs_len):
                    self._single_frame[i] = self._state[i]
                    i += 1
            else:
                for j in range(obs_len):
                    self._single_frame[i] = obs_vector[j] * scale[j]
                    i += 1

        # 2. Shift stacked part (insert newest at the front)
        L = self._single_frame_len
        S = self._mode.stack_size
        if S > 1:
            for k in range(S - 1, 0, -1):
                self._state[k * L:(k + 1) * L] = self._state[(k - 1) * L:k * L]
        self._state[0:L] = self._single_frame

        # 3. Fill non-stacked part
        base = L * S
        for obs_key in self._mode.non_stacked_obs_order:
            obs_len = self._obs_to_length[obs_key]
            if obs_key == "command":
                obs_vector = cmd.get("cmd_vector")
                scale = self._mode.cmd_scale
            elif obs_key == "last_action":
                obs_vector = self._last_action
                scale = self._mode.obs_scale.get(obs_key, [1.0] * obs_len)
            else:
                obs_vector = obs.get(obs_key)
                scale = self._mode.obs_scale.get(obs_key, [1.0] * obs_len)

            if obs_vector is None:
                base += obs_len  # If missing, keep old values
            else:
                for j in range(obs_len):
                    self._state[base + j] = obs_vector[j] * scale[j]
                base += obs_len

        return self._state.tolist()

    def select_action(self, state: list) -> List[float]:
        self._ensure_mode()
        action = self._mode.policy.inference(state)

        for i in range(self._obs_to_length["last_action"]):
            self._scaled_action[i] = action[i] * self._mode.action_scale[i]
        self._last_action = action
        return self._scaled_action.tolist()
