from w4_sdk import *

# Robot's Gains
kp = [100, 100, 100, 100, 120, 120, 0, 0, 100, 100, 100, 100, 120, 120, 0, 0]
kd = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.7, 0.7, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.7, 0.7]

# RL mode
mode = Mode(mode_cfg={
    "id" : 1,
    "stacked_obs_order": ["dof_pos", "dof_vel", "ang_vel", "proj_grav", "last_action"],
    "non_stacked_obs_order": ["command"],
    "obs_scale": {"dof_vel": 0.15,
                  "ang_vel": 0.25,
                  "command": [2.0, 1.0, 0.25]},
    "action_scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 20.0],
    "stack_size": 3,
    "policy_path": "weight/4w4l_mlp_policy.onnx",
    "cmd_vector_length": 3,
})

# Instances
robot = Robot()
joystick = Joystick(max_cmd=[1.0, 0, 1.0])
rl = RL()

# Set gains
robot.set_gains(kp=kp, kd=kd)

# Add & Set Mode
rl.add_mode(mode)
rl.set_mode(mode_id=1)

@control_rate(robot, hz=50)
def loop():
    obs = robot.get_obs()             # Get observation
    print(obs)
    cmd = joystick.get_cmd()          # Get command

    state = rl.build_state(obs, cmd)  # Build state
    action = rl.select_action(state)  # Select action
    action = [0] * 16
    #robot.do_action(action)           # Do action

loop()

