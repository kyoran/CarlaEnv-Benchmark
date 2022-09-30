[repo-name]:CarlaEnv-Benchmark

# CarlaEnv-Benchmark

An open source benchmark for (multi-task) reinforcement learning of autonomous vehicles, and world modeling of intelligent transportation system, etc.


## Contents

- Requirements
- Installation
- Usage
- Custom assets
- Citation
- Acknowledgements

## Requirements

[repo-name] is developed and tested under the following settings:

- **Ubuntu**: 18.04
- **Carla**: 0.9.13
- **Python**: 3.7.13

## Installation

- Download source code of [repo-name]:
```bash
git clone https://github.com/kyoran/CarlaEnv-Benchmark [repo-name]
```

- Installing the Python API of Carla:
```bash
cd .../carla_root/PythonAPI/carla/dist
easy_install carla-0.9.13-py3.7-linux-x86_64.egg
```

- Installing necessary packages:
```bash
pip install -r requirements.txt
```

## Usage

- To manipulate Carla Environment through the Python API, we first need to add the following code to the start of the script or the main function:
```python
import json
import matplotlib.pyplot as plt

from env.CarlaEnv import CarlaEnv
from utils.VideoRecorder import VideoRecorder
```

- We also need to load config files:
```python
with open('./cfg/weather.json', 'r', encoding='utf8') as fff:
    weather_params = json.load(fff)
with open('./cfg/scenario.json', 'r', encoding='utf8') as fff:
    scenario_params = json.load(fff)
```

- Then, we can create a carla environment with selected weather "hard_high_light" and selected scenario "jaywalk":
```python
carla_env = CarlaEnv(
    weather_params=weather_params,
    scenario_params=scenario_params,
    selected_weather="hard_high_light",
    selected_scenario="jaywalk",
    carla_rpc_port=12321,
    carla_tm_port=18935,
    carla_timeout=8,
    ego_auto_pilot=True,   # testing purpose
    perception_type="dvs+vidar",
    num_cameras=5,
    rl_image_size=256,
    fov=60,
    max_fps=120,
    min_fps=30,
    max_episode_steps=1000,
    frame_skip=1,
)
```

- To record what has happened in the process of simulation, we can create a recorder:
```python
video = VideoRecorder("./video", min_fps=30, max_fps=120)
```

- Now the simulation is running by using the following code, with numerous vehicles driving around the map, several pedestrians jaywalking, and a third-person-perspective camera recording data, and five first-person perspective perception data from the ego vehicle. This data can then be used to feed a machine learning algorithm for training an autonomous driving agent.
```
obs = carla_env.reset()
video.init(True)
for one_step in range(400):
    action = [0, 0.7]
    obs, reward, done, info = carla_env.step(action)

    video.record(obs, carla_env.vehicle)
video.save("test")
```

Finally, we can get recorded video in the 'video' directory.

## Settings


## Citation
Publications using [repo-name] are recorded in [Publications]. If you use [repo-name] in your paper, you can also add it to this table by pull request.

If you use SpikingJelly in your work, please cite it as follows:
@misc{[repo-name],
    title = {[repo-name]},
    author = {Xu, Haoran and Chen, Ding and other contributors},
    year = {2020},
    howpublished = {\url{https://github.com/kyoran/CarlaEnv-Benchmark}},
    note = {Accessed: YYYY-MM-DD},
}


## Acknowledgements
Sun Yat-Sen University, Shanghai Jiao Tong University, and Peng Cheng Laboratory are the main developers of Carla-Benchmark.

The list of developers can be found [here](https://github.com/kyoran/CarlaEnv-Benchmark/graphs/contributors).
