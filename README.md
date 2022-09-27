# CarlaEnv-Benchmark

An open source benchmark for (multi-task) reinforcement learning of autonomous vehicles.


## Contents

- Requirements
- Installation
- Usage
- Settings
- Citation
- Acknowledgements

## Requirements

- **Carla**: 0.9.13
- **Python**: >= 3.7

## Installation

- Installing the Python API of Carla:
```bash
cd .../carla_root/PythonAPI/carla/dist
easy_install carla-0.9.13-py3.7-linux-x86_64.egg
```

- Installing necessary packages:
```bash
pip install requirements.txt
```

## Usage

- To manipulate Carla Environment through the Python API, we first need to add the following code to the start of the script or the main function:
```python
import json
import matplotlib.pyplot as plt

from env.CarlaEnv import CarlaEnv
from utils.VideoRecorder import VideoRecorder
```

- reading config files:
```python
with open('./cfg/weather.json', 'r', encoding='utf8') as fff:
    weather_params = json.load(fff)
with open('./cfg/scenario.json', 'r', encoding='utf8') as fff:
    scenario_params = json.load(fff)
```

- creating a carla environment:
```python
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
    max_fps=30,
    min_fps=30,
    max_episode_steps=1000,
    frame_skip=1,
)
```

- creating a recorder:
```python
video = VideoRecorder("./video", fps=30)
```

- testing and recording environment:
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


## Acknowledgements

The list of developers can be found [here](https://github.com/kyoran/CarlaEnv-Benchmark/graphs/contributors).
