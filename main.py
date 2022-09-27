import carla

import json
import time
import matplotlib.pyplot as plt


from env.CarlaEnv import CarlaEnv
from utils.VideoRecorder import VideoRecorder



if __name__ == '__main__':


    # [1] reading cfg
    with open('./cfg/weather.json', 'r', encoding='utf8') as fff:
        weather_params = json.load(fff)
    with open('./cfg/scenario.json', 'r', encoding='utf8') as fff:
        scenario_params = json.load(fff)

    fps = 30

    # [2] creating env
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

    # [3] creating recorder
    video = VideoRecorder("./video", fps=fps)

    # [4] testing and recording env
    max_episode_num = 100
    max_step_num = 100

    for one_episode in range(max_episode_num):
        
        try:
            obs = carla_env.reset()

            print("starting episode:", one_episode+1, "init-frame:", carla_env.frame)

            video.init(True) 
            for one_step in range(max_step_num):

                if one_step <= 50:
                    action = [0, 0.7]
                else:
                    action = [0, -0.3]

                obs, reward, done, info = carla_env.step(action)

                video.record(obs, carla_env.vehicle)

                print(f"\r\tstep: {one_step+1}, frame: {carla_env.frame}", end="")

        except Exception as e:
            print(e)
            
        video.save(f"test-{one_episode+1}")
        print(f"\nsave video done.\n")
