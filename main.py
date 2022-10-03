import carla

import json
import time
import matplotlib.pyplot as plt


from env.CarlaEnv import CarlaEnv
from utils.VideoRecorder import VideoRecorder

"""

"""


if __name__ == '__main__':


    # [1] reading cfg
    with open('./cfg/weather.json', 'r', encoding='utf8') as fff:
        weather_params = json.load(fff)
    with open('./cfg/scenario.json', 'r', encoding='utf8') as fff:
        scenario_params = json.load(fff)

    max_fps = 120
    min_fps = 40


    # [2] creating env
    seletcted_weather = "soft_high_light"
    seletcted_scenario = "jaywalk"

    carla_env = CarlaEnv(
        weather_params=weather_params,
        scenario_params=scenario_params,
        selected_weather=seletcted_weather,
        selected_scenario=seletcted_scenario,
        carla_rpc_port=12321,
        carla_tm_port=18935,
        carla_timeout=8,
        ego_auto_pilot=True,
        perception_type="dvs+vidar",
        # perception_type="rgb",
        num_cameras=5,
        rl_image_size=256,
        fov=60,
        max_fps=max_fps,
        min_fps=min_fps,
        max_episode_steps=1000,
        frame_skip=1,
    )

    # [3] creating recorder
    video = VideoRecorder("./video", min_fps=min_fps, max_fps=max_fps)

    # [4] testing and recording env
    max_episode_num = 1
    max_step_num = 1800

    for one_episode in range(max_episode_num):
        
        try:
            obs = carla_env.reset(seed=19961110)

            print("starting episode:", one_episode+1, "init-frame:", carla_env.frame)

            video.init(True) 
            for one_step in range(max_step_num):

                # if one_step <= max_step_num//2:
                #     action = [0, 0.7]
                # else:
                #     action = [0, -0.3]

                obs, reward, done, info = carla_env.step(action=None)
                """
                obs = {
                    "video_frame": np.array(....),   # shape: 600, 800, 3
                    "rgb_frame": np.array(....),     # shape: 128, 128*5, 3
                }
                """

                video.record(obs, carla_env.vehicle)

                print(f"\r\tstep: {one_step+1}, frame: {carla_env.frame}", end="")

        except Exception as e:
            print(e)
            
        video.save(f"{seletcted_scenario}-{seletcted_weather}:{one_episode+1}", type="mp4")
        # video.save(f"{seletcted_scenario}-{seletcted_weather}:{one_episode+1}", type="gif")

        print(f"\nsave video done.\n")
