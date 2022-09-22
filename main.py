
def main():
    # read config files
    with open('./cfg/weather.json', 'r', encoding='utf8') as fff:
        weather_params = json.load(fff)
    with open('./cfg/scenario.json', 'r', encoding='utf8') as fff:
        scenario_params = json.load(fff)

    # running carla env
    carla_env = CarlaEnv(
        weather_params=weather_params,
        scenario_params=scenario_params,
        selected_weather="L1",
        selected_scenario="tunnel",
        carla_rpc_port=12321,
        carla_tm_port=18935,
        carla_timeout=8,
        perception_type="dvs_frame",
        num_cameras=5,
        rl_image_size=84,
        fov=60,
        max_fps=120,
        min_fps=45,
        max_episode_steps=1000,
        frame_skip=1,
    )
	

if __name__ == '__main__':
	main()