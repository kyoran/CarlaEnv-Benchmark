# CarlaEnv-Benchmark

An autopilot benchmark for multi-modality visual reinforcement learning, and world modeling of intelligent transportation system, etc.

**Note:** Our benchmark is constantly being upgraded, including the addition of extreme weather and challenging traffic scenarios.


<div align="center">
<table border="0" border-collapse="collapse">
    <tr>
        <td>Weather/Scenario</td>
        <td>highway</td>
        <td>narrow</td>
        <td>jaywalk</td>
        <td>tunnel</td>
        <td>merging</td>
    </tr>
    <tr>
        <td>hard_high_light</td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/highway-hard_high_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/narrow-hard_high_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/jaywalk-hard_high_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/tunnel-hard_high_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/crash-hard_high_light.gif" width="120" height="90"/></td>
    </tr>
    <tr>
        <td>soft_high_light</td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/highway-soft_high_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/narrow-soft_high_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/jaywalk-soft_high_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/tunnel-soft_high_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/crash-soft_high_light.gif" width="120" height="90"/></td>
    </tr>
    <tr>
        <td>soft_low_light</td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/highway-soft_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/narrow-soft_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/jaywalk-soft_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/tunnel-soft_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/crash-soft_low_light.gif" width="120" height="90"/></td>
    </tr>
    <tr>
        <td>hard_low_light</td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/highway-hard_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/narrow-hard_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/jaywalk-hard_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/tunnel-hard_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/crash-hard_low_light.gif" width="120" height="90"/></td>
    </tr>
    <tr>
        <td>soft_noisy_low_light</td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/highway-soft_noisy_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/narrow-soft_noisy_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/jaywalk-soft_noisy_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/tunnel-soft_noisy_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/crash-soft_noisy_low_light.gif" width="120" height="90"/></td>
    </tr>
    <tr>
        <td>hard_noisy_low_light</td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/highway-hard_noisy_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/narrow-hard_noisy_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/jaywalk-hard_noisy_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/tunnel-hard_noisy_low_light.gif" width="120" height="90"/></td>
        <td align="center"><img src="https://github.com/kyoran/CarlaEnv-Benchmark/blob/main/img/crash-hard_noisy_low_light.gif" width="120" height="90"/></td>
    </tr>
</table>    
</div>

## Contents

1. Requirements
2. Installation
3. Usage
4. Custom Settings
5. Citation
6. Acknowledgements

## 1/Requirements

`CarlaEnv-Benchmark` is developed and tested under the following settings:

- **Ubuntu**: 18.04
- **Carla**: 0.9.13
- **Python**: 3.7.13

## 2/Installation

- Download the compiled release version and additional maps of CARLA 0.9.13 from [here](https://github.com/carla-simulator/carla/releases/tag/0.9.13) or using the following code:
```shell
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
mkdir CARLA_0.9.13
tar -zxvf CARLA_0.9.13.tar.gz -C CARLA_0.9.13

cd CARLA_0.9.13/
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


We provide five scenarios and six weathers:

### 4.1/Scenario

- ***highway:***
is a wide road with four lanes that vehicles on it run relatively faster than the other scenarios.

- ***narrow:***
is a long and narrow S-bend road that vehicles are forbidden to change lanes.

- ***jaywalk:***
is an intra-city road that some walkers randomly cross.

- ***tunnel:***
is a tunnel where challenging illumination inevitably happens when entering and exiting.

- ***merging:***
is a four-lane road that has three lanes randomly blocked by crashed vehicles.



### 4.2/Weather

- ***hard_high_light:***
is the brightest weather of the midday.

- ***soft_high_light:***
is the twilight when the sun is about to go down.

- ***soft_low_light:***
is a cloudy and foggy day.

- ***hard_low_light:***
is the darkest weather of the midnight.

- ***soft_noisy_low_light:***
is the dust with little rain.

- ***hard_noisy_low_light:***
rain cats and dogs in the midnight.
