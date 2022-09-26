import carla

import json
import matplotlib.pyplot as plt


from env.CarlaEnv import CarlaEnv
from utils.VideoRecorder import VideoRecorder

with open('./cfg/weather.json', 'r', encoding='utf8') as fff:
    weather_params = json.load(fff)
with open('./cfg/scenario.json', 'r', encoding='utf8') as fff:
    scenario_params = json.load(fff)

fps = 20

carla_env = CarlaEnv(
    weather_params=weather_params,
    scenario_params=scenario_params,
    selected_weather="L1",
    selected_scenario="jaywalk",
    carla_rpc_port=12321,
    carla_tm_port=18935,
    carla_timeout=8,
    perception_type="dvs_frame",
    num_cameras=5,
    rl_image_size=128,
    fov=60,
    max_fps=fps,
    min_fps=fps,
    max_episode_steps=1000,
    frame_skip=1,
)

max_episode_num = 100
max_step_num = 200

video = VideoRecorder("./video", fps=fps)

for one_episode in range(max_episode_num):
    
    try:
        print("starting episode:", one_episode+1)
        carla_env.reset()

        video.init(True) 
        for one_step in range(max_step_num):

            if one_step <= 100:
                action = [0, 0.7]
            else:
                action = [0, -0.3]

            obs, reward, done, info = carla_env.step(action)

            video.record(obs, carla_env.vehicle)


            print(f"\rframe: {carla_env.frame}", end="")

    except Exception as e:
        print(e)
        
    video.save(f"test-{one_episode+1}")
    print(f"\nsave video done.\n")

    
