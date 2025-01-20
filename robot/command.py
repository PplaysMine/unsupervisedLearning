from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import matplotlib.pyplot as plt
import os


# directory to save data
output_dir = "./recorded_data"
os.makedirs(output_dir, exist_ok=True)

# run environment
unity_env = UnityEnvironment("./square_env/SquareRoom.exe")
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

# parameters for cropping
crop_h = 20
crop_w = 10
dim_step = 1

commands = [
    [0.0, 0.0, -0.5 * np.pi],
    [0.0, 0.0, -0.5 * np.pi],
    [0.0, 0.0, -0.5 * np.pi],
    [10.0, 0.0, 0.0],
    [-5.0, 10.0, 0.0],
    [0.0, 0.0, 0.5 * np.pi],
    [2.0, 4.0, 1.0 * np.pi],
    [2.0, 0.0, 1.0 * np.pi],
    [0.0, -2.0, 1.5 * np.pi],
    [0.0, 0.0, -0.5 * np.pi],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -0.5 * np.pi],
    [0.0, 0.0, -0.5 * np.pi],
    [-3.0, -1.0, 1.5 * np.pi],
    [1.0, -0.5, 0.0],
    [1.0, -0.5, 0.0],
    [0.0, 0.0, -0.5 * np.pi],
    [0.0, 0.0, -0.5 * np.pi],
    [-3.0, 0.0, 0.0],
    [0.0, 0.0, -0.5 * np.pi]
]

for step, command in enumerate(commands):
    step_output = env.step(command)
    obs_ego, _, vectorial = step_output[0]
    
    print(f"Step {step}: obs_ego shape: {obs_ego.shape}")
    print(f"Step {step}: vectorial: {vectorial}")
    collision_status = vectorial[0]
    print(f"collision {collision_status}")

    data = obs_ego[crop_h:-crop_h, crop_w:-crop_w, :]  # Crop
    data = data[::dim_step, ::dim_step, :]  # Downsample

    # save camera and positional data
    np.save(f"{output_dir}/camera_step_{step}.npy", data)
    np.save(f"{output_dir}/position_step_{step}.npy", vectorial)

print(f"Data recording completed. Files are saved in {output_dir}.")
