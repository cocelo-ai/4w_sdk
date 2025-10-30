#!/usr/bin/env bash

# --- Pick Python with conda priority ---
if [[ -n "${PYTHON_EXEC:-}" ]]; then
  PYTHON="${PYTHON_EXEC}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  PYTHON="$(command -v python3)"
fi

# ===== Smoke test =====
echo "== Smoke test =="
"${PYTHON}" - <<'PY'
import os, random, sys
print("Python:", sys.executable)

# --- Smoke test (onnxpolicy) ---------------------------------------------
from w4_sdk import *

pol = MLPPolicy("weight/4w4l_mlp_policy.onnx") # no_arg test
obs = [random.uniform(-0.5, 0.5) for _ in range(153)]
mlp_pol = MLPPolicy(path="weight/4w4l_mlp_policy.onnx") # arg test
lstm_pol = MLPPolicy(path="weight/4w4l_mlp_policy.onnx")

action = mlp_pol.inference(obs)   # mlp policy inference test
print("action:", action)
action = mlp_pol.inference(obs)   # mlp policy 2nd-inference test
print("action:", action)
print()
action = lstm_pol.inference(obs)   # lstm policy inference test
print("action:", action)
action = lstm_pol.inference(obs)   # lstm policy 2nd-inference test
print("action:", action)

print("=========================================================")
print("============    import onnxpolicy ... OK!    ============")
print("=========================================================\n")

# --- Smoke test (mode) ---------------------------------------------
mode = Mode(mode_cfg={
    "id" : 1,
    "stacked_obs_order": ["dof_pos", "dof_vel", "ang_vel", "proj_grav", "last_action"],
    "non_stacked_obs_order": ["command"],
    "obs_scale": {"dof_vel": 0.15,
                  "ang_vel": 0.25,
                  "command": [2.0, 1.0, 0.25]},
    "action_scale": [1.0]*16,
    "stack_size": 3,
    "policy_path": "weight/4w4l_mlp_policy.onnx",
    "cmd_vector_length": 3,
})
import numpy as np
obs = [random.uniform(-0.5, 0.5) for _ in range(153)]
action = mode.policy.inference(obs)  # only accept  1D list/array
print("len(action):", len(action))
print("type(action)", type(action))

print("=========================================================")
print("===============    import mode ... OK!    ===============")
print("=========================================================\n")

rl = RL()
rl.add_mode(mode)
rl.set_mode(mode_id=1)
obs1 = {
        "dof_pos": [0.0] * 12,
        "dof_vel": [0.0] * 16,
        "ang_vel": [0.0] * 3,
        "proj_grav": [0.0] * 3,
        "last_action": [0.0] * 16,
        "height_map": [0.6128] * 144,
}
obs2 = {
        "dof_pos": [1.0] * 12,
        "dof_vel": [1.0] * 16,
        "ang_vel": [1.0] * 3,
        "proj_grav": [1.0] * 3,
        "last_action": [1.0] * 16,
        "height_map": [1.6128] * 144,
}
obs3 = {
        "dof_pos": [2.0] * 12,
        "dof_vel": [2.0] * 16,
        "ang_vel": [2.0] * 3,
        "proj_grav": [2.0] * 3,
        "last_action": [2.0] * 16,
        "height_map": [2.6128] * 144,
}

obs4 = {
        "dof_pos": [3.0] * 12,
        "dof_vel": [3.0] * 16,
        "ang_vel": [3.0] * 3,
        "proj_grav": [3.0] * 3,
        "last_action": [3.0] * 16,
}

obs5 = {
}


cmd = {"cmd_vector": [0, 0, 0]}
state1 = rl.build_state(obs1, cmd)
action = rl.select_action(state1)
state2 = rl.build_state(obs2, cmd)
action = rl.select_action(state2)
state3 = rl.build_state(obs3, cmd)
action = rl.select_action(state3)
state4 = rl.build_state(obs4, cmd)
action = rl.select_action(state4)
state5 = rl.build_state(obs5, cmd, scaled_last_action = [9] * 16)
action = rl.select_action(state5)
print("state1", state1, "\n\n")
print("state2", state2, "\n\n")
print("state3", state3, "\n\n")
print("state4", state4, "\n\n")
print("state5", state5, "\n\n")
print("=========================================================")
print("===============    import rl ... OK!    ===============")
print("=========================================================\n")

robot = Robot()
obs = robot.get_obs()
print("obs:", obs)
action = [0] * 8
robot.do_action(action)
robot.estop()
print("=========================================================")
print("===============    import robot ... OK!    ===============")
print("=========================================================\n")
PY
