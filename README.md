# CarlaEnv-Benchmark

An open source benchmark for (multi-task) reinforcement learning of autonomous vehicles, and world modeling of intelligent transportation system, etc.


## Contents

- 1. Requirements
- 2. Installation
- 3. Usage
- 4. Custom Settings
- 5. Citation
- 6. Acknowledgements

## 1/Requirements

`CarlaEnv-Benchmark` is developed and tested under the following settings:

- **Ubuntu**: 18.04
- **Carla**: 0.9.13
- **Python**: 3.7.13

## 2/Installation

- Download the compiled release version and additional maps of CARLA 0.9.13 from [here](https://github.com/carla-simulator/carla/releases/tag/0.9.13) or using the following code:
```shell
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz -o carla_0.9.13.tar.gz
tar -zxvf carla_0.9.13.tar.gz

cd carla_0.9.13/
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz
tar -zxvf AdditionalMaps_0.9.13.tar.gz
```


- Installing the Python API of Carla:
```bash
cd carla_0.9.13/PythonAPI/carla/dist
easy_install carla-0.9.13-py3.7-linux-x86_64.egg
```


- Download source code of `CarlaEnv-Benchmark`:
```bash
git clone https://github.com/kyoran/CarlaEnv-Benchmark
```


- Installing necessary packages:
```bash
pip install -r requirements.txt
```

## 3/Usage

- We first need to run the rendering engine CARLA server in the background:
```shell
cd carla_0.9.13/
DISPLAY= ./CarlaUE4.sh -opengl -RenderOffScreen -world-port=12321
```


- To manipulate the environment through the Python API, we need to add the following code to the start of the script or the main function:
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
    perception_type="dvs+vidar",
    num_cameras=5,
    rl_image_size=256,
    fov=60,
    max_fps=120,
    min_fps=30,
    max_episode_steps=1000,
    frame_skip=1,
    ego_auto_pilot=True,   # testing purpose
    is_spectator=True,     # rendering mode
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

## 4/Custom Settings

### Scenario
We provide five typical scenarios:
- ***highway***
- ***narrow***
- ***jaywalk***
- ***tunnel*** 
- ***merging***

### Weather


## 5/Citation
Publications using `CarlaEnv-Benchmark` are recorded in [Publications]. If you use [repo-name] in your paper, you can also add it to this table by pull request.

If you use SpikingJelly in your work, please cite it as follows:
```latex
@misc{CarlaEnv-Benchmark,
    title = {CarlaEnv-Benchmark},
    author = {Xu, Haoran and Chen, Ding and other contributors},
    year = {2022},
    howpublished = {\url{https://github.com/kyoran/CarlaEnv-Benchmark}},
    note = {Accessed: YYYY-MM-DD},
}
```

***Note***: To specify the version of `CarlaEnv-Benchmark` you are using, the default value `YYYY-MM-DD` in the note field should be replaced with the date of the last change of the framework you are using, i.e. the date of the latest commit.

## 6/Acknowledgements
Sun Yat-Sen University, Shanghai Jiao Tong University, Peking University, and Peng Cheng Laboratory are the main developers of `CarlaEnv-Benchmark`.


<div align="center">
<table>
    <tr>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/sysu_logo.png" width="80" height="80" alt="sysu" /></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/sjtu_logo.png" width="80" height="80" alt="sjtu" /></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/pku_logo.png" width="80" height="80" alt="pku" /></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/pcnl_logo.png" width="80" height="80" alt="pcnl" /></td>
    </tr>
</table>
</div>


The list of developers can be found [here](https://github.com/kyoran/CarlaEnv-Benchmark/graphs/contributors).
