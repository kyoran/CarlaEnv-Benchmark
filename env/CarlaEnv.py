# !/usr/bin/python3
# -*- coding: utf-8 -*-
# CARLA 0.9.13 environment

"""
sudo docker run --privileged --user carla --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla-add:0.9.13 /bin/bash ./CarlaUE4.sh -world-port=12321 -RenderOffScreen
"""

import carla

import os
import sys
import time
import math
import random
import numpy as np
from dotmap import DotMap


class CarlaEnv(object):

    def __init__(self,
                 weather_params, scenario_params,
                 selected_weather, selected_scenario,
                 carla_rpc_port, carla_tm_port, carla_timeout,
                 perception_type, num_cameras, rl_image_size, fov,
                 max_fps, min_fps, control_hz,
                 max_episode_steps, frame_skip,
                 is_spectator=False, ego_auto_pilot=False,
                 dvs_rec_args=None,
                 ):

        self.frame_skip = frame_skip

        self.carla_rpc_port = carla_rpc_port
        self.carla_tm_port = carla_tm_port
        self.carla_timeout = carla_timeout
        self.weather_params = weather_params
        self.scenario_params = scenario_params

        # testing params
        self.ego_auto_pilot = ego_auto_pilot
        self.is_spectator = is_spectator

        self.num_cameras = num_cameras
        self.rl_image_size = rl_image_size
        self.fov = fov
        self.max_fps = max_fps
        self.min_fps = min_fps
        self.control_hz = control_hz
        self.max_episode_steps = 20 * self.max_fps

        self.selected_weather = selected_weather
        self.selected_scenario = selected_scenario

        # rgb-frame, dvs-rec-frame, dvs-stream, dvs-vidar-stream
        self.perception_type = perception_type  # ↑↑↑↑↑↑↑↑↑
        if self.perception_type == "dvs-rec-frame":
            assert dvs_rec_args, "missing necessary param: [dvs_rec_args]"

            self.dvs_rec_args = dvs_rec_args

            sys.path.append("./tools/rpg_e2vid")
            # from tools.rpg_e2vid.run_dvs_rec import run_dvs_rec
            # from run_dvs_rec import run_dvs_rec
            # from tools.rpg_e2vid.e2vid_utils.loading_utils import load_model, get_device
            from e2vid_utils.loading_utils import load_model, get_device

            # Load model
            self.rec_model = load_model(self.dvs_rec_args.path_to_model)
            self.device = get_device(self.dvs_rec_args.use_gpu)
            self.rec_model = self.rec_model.to(self.device)
            self.rec_model .eval()

        elif self.perception_type == "vidar-rec-frame":
            sys.path.append("./tools/vidar2frame")
            # vidar reconstruction do not need to load a model


        # client init
        self.client = carla.Client('localhost', self.carla_rpc_port)
        self.client.set_timeout(self.carla_timeout)

        # world
        self.world = self.client.load_world(self.scenario_params[self.selected_scenario]["map"])

        assert self.client.get_client_version() == "0.9.13"
        assert self.selected_scenario in self.scenario_params.keys()
        assert self.selected_weather in self.weather_params.keys()

        #
        self.vehicle_actors = []
        self.sensor_actors = []
        self.walker_ai_actors = []
        self.walker_actors = []

        self.reset_num = 0

        # reset
        self.reset()

    def _init_blueprints(self):

        self.bp_lib = self.world.get_blueprint_library()

        self.collision_bp = self.bp_lib.find('sensor.other.collision')

        self.video_camera_bp = self.bp_lib.find('sensor.camera.rgb')

        self.rgb_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('sensor_tick', f'{1 / self.min_fps}')
        self.rgb_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.rgb_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        self.rgb_camera_bp.set_attribute('fov', str(self.fov))
        self.rgb_camera_bp.set_attribute('enable_postprocess_effects', str(True))

        self.dvs_camera_bp = self.bp_lib.find('sensor.camera.dvs')
        self.dvs_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
        #         dvs_camera_bp.set_attribute('positive_threshold', str(0.3))
        #         dvs_camera_bp.set_attribute('negative_threshold', str(0.3))
        #         dvs_camera_bp.set_attribute('sigma_positive_threshold', str(0))
        #         dvs_camera_bp.set_attribute('sigma_negative_threshold', str(0))
        self.dvs_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.dvs_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        self.dvs_camera_bp.set_attribute('fov', str(self.fov))
        self.dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))


        self.vidar_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.vidar_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
        self.vidar_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.vidar_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        self.vidar_camera_bp.set_attribute('fov', str(self.fov))
        self.vidar_camera_bp.set_attribute('enable_postprocess_effects', str(True))

    def _set_dummy_variables(self):
        # dummy variables given bisim's assumption on deep-mind-control suite APIs
        low = -1.0
        high = 1.0
        self.action_space = DotMap()
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = [2]
        self.observation_space = DotMap()
        # D, H, W
        if self.perception_type.__contains__("frame"):
            self.observation_space.shape = (3, self.rl_image_size, self.num_cameras * self.rl_image_size)
            self.observation_space.dtype = np.dtype(np.uint8)
        if self.perception_type.__contains__("rec"):
            self.observation_space.shape = (1, self.rl_image_size, self.num_cameras * self.rl_image_size)
            self.observation_space.dtype = np.dtype(np.uint8)
        if self.perception_type.__contains__("stream"):
            self.observation_space.shape = (None, 4)
            self.observation_space.dtype = np.dtype(np.float32)
        self.reward_range = None
        self.metadata = None
        self.action_space.sample = lambda: np.random.uniform(
            low=low, high=high, size=self.action_space.shape[0]).astype(np.float32)

    def _dist_from_center_lane(self, vehicle, info):
        # assume on highway
        vehicle_location = vehicle.get_location()
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_s = vehicle_waypoint.s
        vehicle_velocity = vehicle.get_velocity()  # Vecor3D
        vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
        speed = np.linalg.norm(vehicle_velocity_xy)

        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        assert road_id is not None
        lane_id = int(vehicle_waypoint_closest_to_road.lane_id)
        goal_lane_id = lane_id

        current_waypoint = self.map.get_waypoint(vehicle_location, project_to_road=False)
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        if goal_waypoint is None:
            # try to fix, bit of a hack, with CARLA waypoint discretizations
            carla_waypoint_discretization = 0.02  # meters
            goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s - carla_waypoint_discretization)
            if goal_waypoint is None:
                goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id,
                                                           vehicle_s + carla_waypoint_discretization)

        if goal_waypoint is None:
            print("Episode fail: goal waypoint is off the road! (frame %d)" % self.time_step)
            done, dist, vel_s = True, 100., 0.
            info['reason_episode_ended'] = 'off_road'

        else:
            goal_location = goal_waypoint.transform.location
            goal_xy = np.array([goal_location.x, goal_location.y])
            dist = np.linalg.norm(vehicle_xy - goal_xy)

            next_goal_waypoint = goal_waypoint.next(0.1)  # waypoints are ever 0.02 meters
            if len(next_goal_waypoint) != 1:
                print('warning: {} waypoints (not 1)'.format(len(next_goal_waypoint)))
            if len(next_goal_waypoint) == 0:
                print("Episode done: no more waypoints left. (frame %d)" % self.time_step)
                info['reason_episode_ended'] = 'no_waypoints'
                done, vel_s = True, 0.
            else:
                location_ahead = next_goal_waypoint[0].transform.location
                highway_vector = np.array([location_ahead.x, location_ahead.y]) - goal_xy
                highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)
                done = False

        # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        #         if vehicle_velocity.z > 1. and self.time_step < 20:
        #             print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_velocity.z, self.time_step))
        #             done = True
        #         if vehicle_location.z > 0.5 and self.time_step < 20:
        #             print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_location.z, self.time_step))
        #             done = True

        return dist, vel_s, speed, done

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        # print('Collision (intensity {})'.format(intensity))
        self._collision_intensities_during_last_time_step.append(intensity)

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
            filt: the filter indicating what type of actors we'll look at.

        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly

        return actor_poly_dict

    def _control_all_walkers(self):

        walker_behavior_params = self.scenario_params[self.selected_scenario]["walker_behavior"]

        # if walker is dead 
        # for walker in self.walker_actors:
        #     if not walker.is_alive:
        #         walker.destroy()
        #         self.walker_actors.remove(walker)


        all_veh_locs = [
            [one_actor.get_transform().location.x, one_actor.get_transform().location.y] 
            for one_actor in self.vehicle_actors
        ]
        all_veh_locs = np.array(all_veh_locs, dtype=np.float32)

        for walker in self.walker_actors:
            if walker.is_alive:
                # get location and velocity of the walker
                loc_x, loc_y = walker.get_location().x, walker.get_location().y
                vel_x, vel_y = walker.get_velocity().x, walker.get_velocity().y
                walker_loc = np.array([loc_x, loc_y], dtype=np.float32)

                # judge whether walker can cross the road
                dis_gaps = np.linalg.norm(all_veh_locs - walker_loc, axis=1)
                cross_flag = (dis_gaps >= walker_behavior_params["secure_dis"]).all()
                cross_prob = walker_behavior_params["cross_prob"]

                if loc_y > walker_behavior_params["border"]["y"][1]:
                    if self.time_step % self.max_fps == 0 and random.random() < cross_prob and cross_flag:
                        walker.apply_control(self.left)
                    else:
                        if loc_x > walker_behavior_params["border"]["x"][1]:
                            walker.apply_control(self.backward)

                        elif loc_x > walker_behavior_params["border"]["x"][0]:
                            if vel_x > 0:
                                walker.apply_control(self.forward)
                            else:
                                walker.apply_control(self.backward)

                        else:
                            walker.apply_control(self.forward)
                    
                elif loc_y > walker_behavior_params["border"]["y"][0] and cross_flag:
                    if vel_y > 0:
                        walker.apply_control(self.right)
                    else:
                        walker.apply_control(self.left)
                    
                else:
                    if self.time_step % self.max_fps == 0 and random.random() < cross_prob and cross_flag:
                        walker.apply_control(self.right)

                    else:
                        if loc_x > walker_behavior_params["border"]["x"][1]:
                            walker.apply_control(self.backward)

                        elif loc_x > walker_behavior_params["border"]["x"][0]:
                            if vel_x > 0:
                                walker.apply_control(self.forward)
                            else:
                                walker.apply_control(self.backward)

                        else:
                            walker.apply_control(self.forward)

    def _clear_all_actors(self):
        # remove all vehicles, walkers, and sensors (in case they survived)
        # self.world.tick()

        if 'vehicle' in dir(self) and self.vehicle is not None:
            for one_sensor_actor in self.sensor_actors:
                if one_sensor_actor.is_alive:
                    one_sensor_actor.stop()
                    one_sensor_actor.destroy()

        # # self.vidar_data['voltage'] = np.zeros((self.obs_size, self.obs_size), dtype=np.uint16)
        for actor_filter in ['vehicle.*', 'walker.*']:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    actor.destroy()

        # for one_vehicle_actor in self.vehicle_actors:
        #     if one_vehicle_actor.is_alive:
        #         one_vehicle_actor.destroy()

        # for one_walker_ai_actor in self.walker_ai_actors:
        #     if one_walker_ai_actor.is_alive:
        #         one_walker_ai_actor.stop()
        #         one_walker_ai_actor.destroy()

        # for one_walker_actor in self.walker_actors:
        #     if one_walker_actor.is_alive:
        #         one_walker_actor.destroy()


        # for actor_filter in ['vehicle.*', 'controller.ai.walker', 'walker.*', 'sensor*']:
        #     for actor in self.world.get_actors().filter(actor_filter):
        #         if actor.is_alive:
        #             if actor.type_id == 'controller.ai.walker':
        #                 actor.stop()
        #             actor.destroy()

        self.vehicle_actors = []
        self.sensor_actors = []
        self.walker_actors = []
        self.walker_ai_actors = []

        # self.world.tick()
        # self.client.reload_world(reset_settings=True)

    def _set_seed(self, seed):
        if seed:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            random.seed(seed)

            self.tm.set_random_device_seed(seed)

    def reset(self, seed=None):

        self._clear_all_actors()

        # self.client.reload_world(reset_settings = True)

        if self.reset_num == 0:

            self._set_dummy_variables()

            # self.world = self.client.load_world(
            #     map_name = self.scenario_params[self.selected_scenario]["map"],
            #     reset_settings = False
            # )

            # remove dynamic objects to prevent 'tables' and 'chairs' flying in the sky
            env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Dynamic)
            objects_to_toggle = set([one_env_obj.id for one_env_obj in env_objs])
            self.world.enable_environment_objects(objects_to_toggle, False)
            self.map = self.world.get_map()

            # bp
            self._init_blueprints()

            # spectator
            if self.is_spectator:
                self.spectator = self.world.get_spectator()
            else:
                self.spectator = None

            # tm
            self.tm = self.client.get_trafficmanager(self.carla_tm_port)
            self.tm_port = self.tm.get_port()
            self.tm.set_global_distance_to_leading_vehicle(2.0)
            
            # lm
            self.lm = self.world.get_lightmanager()

        # 
        self._set_seed(seed)

        # reset
        self.reset_sync_mode(False)

        self.reset_surrounding_vehicles()
        self.reset_special_vehicles()
        self.reset_walkers()
        self.reset_ego_vehicle()

        self.reset_weather()
        self.reset_sensors()

        self.reset_sync_mode(True)

        # spectator
        # if self.spectator is not None:
        #     self.spectator.set_transform(
        #         carla.Transform(self.vehicle.get_transform().location + carla.Location(z=40),
        #         carla.Rotation(pitch=-90)))

        self.time_step = 0
        self.dist_s = 0
        self.return_ = 0
        self.velocities = []

        self.reward = [0]
        self.perception_data = []
        self.last_action = None

        # MUST warm up !!!!!!
        # take some steps to get ready for the dvs+vidar camera, walkers, and vehicles
        obs = None
        warm_up_max_steps = self.control_hz     # 15
        while warm_up_max_steps > 0:
            warm_up_max_steps -= 1
            # self.world.tick()
            obs, _, _, _ = self.step(None)
            # print("len:self.perception_data:", len(self.perception_data))
            
        # self.vehicle.set_autopilot(True, self.carla_tm_port)
        # while abs(self.vehicle.get_velocity().x) < 0.02:
        #     #             print("!!!take one init step", warm_up_max_steps, self.vehicle.get_control(), self.vehicle.get_velocity())
        #     self.world.tick()
        #     #             action = self.compute_steer_action()
        #     #             obs, _, _, _ = self.step(action=action)
        #     #             self.time_step -= 1
        #     warm_up_max_steps -= 1
        #     if warm_up_max_steps < 0 and self.dvs_data['events'] is not None:
        #         break
        # self.vehicle.set_autopilot(False, self.carla_tm_port)

        self.init_frame = self.frame
        self.reset_num += 1
        # print("carla env reset done.")

        return obs

    def reset_sync_mode(self, synchronous_mode=True):

        self.delta_seconds = 1.0 / self.max_fps
        # max_substep_delta_time = 0.005

        #         self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=synchronous_mode,
            fixed_delta_seconds=self.delta_seconds,
            # substepping=True,
            # max_substep_delta_time=0.005,
            # max_substeps=int(self.delta_seconds/max_substep_delta_time)
            ))
        self.tm.set_synchronous_mode(synchronous_mode)

    def reset_surrounding_vehicles(self):
        total_surrounding_veh_num = 0
        veh_bp = self.bp_lib.filter('vehicle.*')
        veh_bp = [x for x in veh_bp if int(x.get_attribute('number_of_wheels')) == 4]

        for one_type in ["same_dir_veh", "oppo_dir_veh"]:
            # for one_type in ["same_dir_veh"]:

            one_type_params = self.scenario_params[self.selected_scenario][one_type]

            # print("now at:", one_type)

            for one_part in range(len(one_type_params)):

                veh_num = one_type_params[one_part]["num"]

                while veh_num > 0:

                    rand_veh_bp = random.choice(veh_bp)

                    spawn_road_id = one_type_params[one_part]["road_id"]
                    spawn_lane_id = random.choice(
                        one_type_params[one_part]["lane_id"])
                    spawn_start_s = np.random.uniform(
                        one_type_params[one_part]["start_pos"][0],
                        one_type_params[one_part]["start_pos"][1],
                    )

                    veh_pos = self.map.get_waypoint_xodr(
                        road_id=spawn_road_id,
                        lane_id=spawn_lane_id,
                        s=spawn_start_s,
                    ).transform
                    veh_pos.location.z += 0.1

                    if rand_veh_bp.has_attribute('color'):
                        color = random.choice(rand_veh_bp.get_attribute('color').recommended_values)
                        rand_veh_bp.set_attribute('color', color)
                    if rand_veh_bp.has_attribute('driver_id'):
                        driver_id = random.choice(rand_veh_bp.get_attribute('driver_id').recommended_values)
                        rand_veh_bp.set_attribute('driver_id', driver_id)
                    rand_veh_bp.set_attribute('role_name', 'autopilot')
                    vehicle = self.world.try_spawn_actor(rand_veh_bp, veh_pos)

                    if vehicle is not None:
                        vehicle.set_autopilot(True, self.tm_port)
                        vehicle.set_light_state(carla.VehicleLightState.HighBeam)

                        self.tm.auto_lane_change(vehicle, True)
                        self.tm.vehicle_percentage_speed_difference(
                            vehicle, one_type_params[one_part]["speed"])
                        self.tm.ignore_lights_percentage(vehicle, 100)
                        self.tm.ignore_signs_percentage(vehicle, 100)
                        self.world.tick()

                        self.vehicle_actors.append(vehicle)

                        veh_num -= 1
                        total_surrounding_veh_num += 1
                        # print(f"\t spawn vehicle: {total_surrounding_veh_num}, at {veh_pos.location}")

    def reset_special_vehicles(self):
        pass

    def reset_walkers(self):
        walker_bp = self.bp_lib.filter('walker.*')
        total_surrounding_walker_num = 0

        walker_params = self.scenario_params[self.selected_scenario]["walker"]
        walker_behavior_params = self.scenario_params[self.selected_scenario]["walker_behavior"]

        self.left = carla.WalkerControl(
            direction=carla.Vector3D(y=-1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))
        self.right = carla.WalkerControl(
            direction=carla.Vector3D(y=1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))

        self.forward = carla.WalkerControl(
            direction=carla.Vector3D(x=1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))
        self.backward = carla.WalkerControl(
            direction=carla.Vector3D(x=-1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))

        for one_part in range(len(walker_params)):

            walker_num = walker_params[one_part]["num"]

            while walker_num > 0:
                rand_walker_bp = random.choice(walker_bp)

                spawn_road_id = walker_params[one_part]["road_id"]
                spawn_lane_id = random.choice(
                    walker_params[one_part]["lane_id"])
                spawn_start_s = np.random.uniform(
                    walker_params[one_part]["start_pos"][0],
                    walker_params[one_part]["start_pos"][1],
                )

                walker_pos = self.map.get_waypoint_xodr(
                    road_id=spawn_road_id,
                    lane_id=spawn_lane_id,
                    s=spawn_start_s,
                ).transform
                walker_pos.location.z += 0.1

                # set as not invencible
                if rand_walker_bp.has_attribute('is_invincible'):
                    rand_walker_bp.set_attribute('is_invincible', 'false')

                walker_actor = self.world.try_spawn_actor(rand_walker_bp, walker_pos)

                if walker_actor:
                    # walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                    # walker_controller_actor = self.world.spawn_actor(
                    #     walker_controller_bp, carla.Transform(), walker_actor)
                    # # start walker
                    # walker_controller_actor.start()
                    # # set walk to random point
                    # #             walker_controller_actor.go_to_location(world.get_random_location_from_navigation())
                    # rand_destination = carla.Location(
                    #     x=np.random.uniform(walker_params[one_part]["dest"]["x"][0], walker_params[one_part]["dest"]["x"][1]),
                    #     y=random.choice([walker_params[one_part]["dest"]["y"][0], walker_params[one_part]["dest"]["y"][1]]),
                    #     z=0.
                    # )
                    # walker_controller_actor.go_to_location(rand_destination)
                    # # random max speed (default is 1.4 m/s)
                    # walker_controller_actor.set_max_speed(
                    #     np.random.uniform(
                    #         walker_params[one_part]["speed"][0],
                    #         walker_params[one_part]["speed"][1]
                    #     ))
                    # self.walker_ai_actors.append(walker_controller_actor)

                    self.walker_actors.append(walker_actor)

                    self.world.tick()
                    walker_num -= 1
                    total_surrounding_walker_num += 1
                    # print(f"\t spawn walker: {total_surrounding_walker_num}, at {walker_pos.location}")

    def reset_ego_vehicle(self):

        self.vehicle = None

        # create vehicle
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        #         print(f"\tlen of self.vehicle_polygons: {len(self.vehicle_polygons[-1].keys())}")
        #         print(self.vehicle_polygons[-1].keys())
        ego_veh_params = self.scenario_params[self.selected_scenario]["ego_veh"]

        ego_spawn_times = 0
        max_ego_spawn_times = 10

        while True:
            # print("ego_spawn_times:", ego_spawn_times)d

            if ego_spawn_times > max_ego_spawn_times:

                ego_spawn_times = 0

                # print("\tspawn ego vehicle times > max_ego_spawn_times")
                self._clear_all_actors()
                self.reset_surrounding_vehicles()
                self.reset_special_vehicles()
                self.reset_walkers()

                self.vehicle_polygons = []
                vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
                self.vehicle_polygons.append(vehicle_poly_dict)

                continue

            # Check if ego position overlaps with surrounding vehicles
            overlap = False

            spawn_road_id = ego_veh_params["road_id"]
            spawn_lane_id = random.choice(ego_veh_params["lane_id"])
            spawn_start_s = np.random.uniform(
                ego_veh_params["start_pos"][0],
                ego_veh_params["start_pos"][1],
            )

            veh_start_pose = self.map.get_waypoint_xodr(
                road_id=spawn_road_id,
                lane_id=spawn_lane_id,
                s=spawn_start_s,
            ).transform
            veh_start_pose.location.z += 0.1


            for idx, poly in self.vehicle_polygons[-1].items():
                poly_center = np.mean(poly, axis=0)
                ego_center = np.array([veh_start_pose.location.x, veh_start_pose.location.y])
                dis = np.linalg.norm(poly_center - ego_center)
                if dis > 8:
                    continue
                else:
                    overlap = True

                    break

            if not overlap:
                self.vehicle = self.world.try_spawn_actor(
                    self.bp_lib.find(ego_veh_params["type"]),
                    veh_start_pose
                )

            if self.vehicle is not None:

                self.vehicle_actors.append(self.vehicle)

                # AUTO pilot
                if self.ego_auto_pilot:
                    self.vehicle.set_autopilot(True, self.tm_port)
                    self.vehicle.set_light_state(carla.VehicleLightState.HighBeam)

                    self.tm.auto_lane_change(self.vehicle, True)
                    self.tm.vehicle_percentage_speed_difference(
                        self.vehicle, ego_veh_params["speed"])
                    self.tm.ignore_lights_percentage(self.vehicle, 100)
                    self.tm.ignore_signs_percentage(self.vehicle, 100)
                else:
                    # immediate running
                    physics_control = self.vehicle.get_physics_control()
                    physics_control.gear_switch_time = 0.01
                    physics_control.damping_rate_zero_throttle_clutch_engaged=physics_control.damping_rate_zero_throttle_clutch_disengaged
                    self.vehicle.apply_physics_control(physics_control)
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1, manual_gear_shift=True, gear=1))
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.01)
                # print("ego_spawn_times:", ego_spawn_times)

        self.world.tick()

    def reset_sensors(self):

        # [one_sensor.stop() for one_sensor in self.sensors]

        # data
        self.video_data = {'frame': 0, 'timestamp': 0.0, 'img': np.zeros((600, 800, 3), dtype=np.uint8)}
        self.rgb_data = {'frame': [0] * self.num_cameras, 'timestamp': [0.0] * self.num_cameras,
                         'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8)}

        if self.perception_type.__contains__("dvs"):
            self.dvs_data = {'frame': [0] * self.num_cameras, 'timestamp': [0.0] * self.num_cameras, 'events': None,
                             'events_tmp': [],
                             'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8),
                             # 'rec-img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 1), dtype=np.uint8),
                             }
        if self.perception_type.__contains__("vidar"):
            self.vidar_data = {'frame': [0] * self.num_cameras, 'timestamp': [0.0] * self.num_cameras,
                               'voltage': np.zeros((self.rl_image_size, self.rl_image_size*self.num_cameras), dtype=np.uint16),
                               'spike': np.zeros((self.rl_image_size, self.rl_image_size*self.num_cameras), dtype=np.bool_),
                               'img': np.zeros((self.rl_image_size, self.rl_image_size*self.num_cameras, 3), dtype=np.uint8),
                               # 'captured_cam': [],
                               # 'rec-img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 1), dtype=np.uint8),
                               }

        self.frame = None

        #         def on_tick_func(data):
        #             self.dvs_data["events"] = None
        #         self.world.on_tick(on_tick_func)

        # Third person perspective
        def __get_video_data__(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.video_data['frame'] = data.frame
            self.video_data['timestamp'] = data.timestamp
            self.video_data['img'] = array

        self.video_camera_rgb = self.world.spawn_actor(
            self.video_camera_bp,
            carla.Transform(carla.Location(x=-5.5, z=3.5), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.video_camera_rgb.listen(lambda data: __get_video_data__(data))
        self.sensor_actors.append(self.video_camera_rgb)

        #         print("\t video sensor init done.")

        # we'll use up to five cameras, which we'll stitch together
        location = carla.Location(x=1.6, z=1.7)

        # Perception RGB sensor
        def __get_rgb_data__(data, one_camera_idx):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.rgb_data['frame'][one_camera_idx] = data.frame
            self.rgb_data['timestamp'][one_camera_idx] = data.timestamp
            self.rgb_data['img'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size, :] = array

        

        self.rgb_camera_left2 = self.world.spawn_actor(
            self.rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=-2 * float(self.fov))),
            attach_to=self.vehicle)
        self.rgb_camera_left1 = self.world.spawn_actor(
            self.rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov))),
            attach_to=self.vehicle)
        self.rgb_camera_mid = self.world.spawn_actor(
            self.rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
            attach_to=self.vehicle)
        self.rgb_camera_right1 = self.world.spawn_actor(
            self.rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov))),
            attach_to=self.vehicle)
        self.rgb_camera_right2 = self.world.spawn_actor(
            self.rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=2 * float(self.fov))),
            attach_to=self.vehicle)

        for one_camera_idx, one_rgb_camera in enumerate([
            self.rgb_camera_left2, self.rgb_camera_left1,
            self.rgb_camera_mid,
            self.rgb_camera_right1, self.rgb_camera_right2
        ]):
            #             print("listen rgb:", one_camera_idx)
            one_rgb_camera.listen(lambda data, one_camera_idx=one_camera_idx: __get_rgb_data__(data, one_camera_idx))

        self.sensor_actors.append(self.rgb_camera_left2)
        self.sensor_actors.append(self.rgb_camera_left1)
        self.sensor_actors.append(self.rgb_camera_mid)
        self.sensor_actors.append(self.rgb_camera_right1)
        self.sensor_actors.append(self.rgb_camera_right2)

        #         print("\t rgb sensors init done.")

        # Perception DVS sensor
        if self.perception_type.__contains__("dvs"):
            def __get_dvs_data__(data, one_camera_idx):
                #             print("get_dvs_data:", one_camera_idx)
                events = np.frombuffer(data.raw_data, dtype=np.dtype([
                    ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool_)]))

                img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
                img[events[:]['y'], events[:]['x'], events[:]['pol'] * 2] = 255
                self.dvs_data['frame'][one_camera_idx] = data.frame
                self.dvs_data['timestamp'][one_camera_idx] = data.timestamp
                self.dvs_data['img'][:, one_camera_idx * self.rl_image_size:
                                        (one_camera_idx + 1) * self.rl_image_size, :] = img

                x = events['x'].astype(np.float64) + one_camera_idx * self.rl_image_size
                y = events['y'].astype(np.float64)
                p = events['pol'].astype(np.float64)
                t = events['t'].astype(np.float64)

                events = np.column_stack((x, y, p, t))  # (event_num, 4)

                if len(self.dvs_data["events_tmp"]) != 5:
                    self.dvs_data["events_tmp"].append(events)
                else:

                    self.dvs_data['events'] = np.concatenate(self.dvs_data["events_tmp"], axis=0)
                    self.dvs_data['events'] = self.dvs_data['events'][np.argsort(self.dvs_data['events'][:, -1])]

                    # rec
                    # if self.perception_type.__contains__("dvs") and self.perception_type.__contains__("rec"):
                    #     from run_dvs_rec import run_dvs_rec
                    #     out = run_dvs_rec(
                    #         # x, y, p, t -> t, x, y, p
                    #         self.dvs_data['events'][:, [3, 0, 1, 2]],
                    #         self.rec_model, self.device, self.dvs_rec_args)
                    #     # print(out.shape, out.dtype)
                    #     # print(out[:5, :5])
                    #     self.dvs_data['rec-img'][:, :, 0] = out

                    # init start, x, y, p, t
                    # (event_num, 4)
                    # x => [0, 1]
                    self.dvs_data['events'][:, 0] = self.dvs_data['events'][:, 0] / (
                                self.rl_image_size * self.num_cameras - 1)
                    # x => [-0.5, 0.5]
                    self.dvs_data['events'][:, 0] -= 0.5
                    # y => [0, 1]
                    self.dvs_data['events'][:, 1] = self.dvs_data['events'][:, 1] / (self.rl_image_size - 1)
                    # y => [-0.5, 0.5]
                    self.dvs_data['events'][:, 1] -= 0.5
                    # p
                    #                 self.dvs_data['events'][:, 2] = self.dvs_data['events'][:, 2]
                    # t
                    t_start = self.dvs_data['events'][0, 3]
                    t_final = self.dvs_data['events'][-1, 3]
                    #                 print("one_obs[3, :].shape:", one_obs[3, :].shape)
                    #                 print("t_start:", t_start, one_obs[3, :].min())
                    #                 print("t_final:", t_final, one_obs[3, :].max())

                    #                 assert t_start == self.dvs_data['events'][:, 3].min() and t_final == self.dvs_data['events'][:, 3].max()

                    dt = t_final - t_start
                    if dt > 0:
                        self.dvs_data['events'][:, 3] = (self.dvs_data['events'][:, 3] - t_start) / dt
                        self.dvs_data['events'][:, 3] = (t_final - self.dvs_data['events'][:, 3]) / dt

                    else:
                        self.dvs_data['events'][:, 3] = 0

                    # init done.
                    self.dvs_data['events'] = self.dvs_data['events'].astype(np.float32)

                    self.dvs_data['events_tmp'].clear()


            self.dvs_camera_left2 = self.world.spawn_actor(
                self.dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov) * 2)), attach_to=self.vehicle)
            self.dvs_camera_left1 = self.world.spawn_actor(
                self.dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov) * 1)), attach_to=self.vehicle)
            self.dvs_camera_mid = self.world.spawn_actor(
                self.dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
            self.dvs_camera_right1 = self.world.spawn_actor(
                self.dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov) * 1)), attach_to=self.vehicle)
            self.dvs_camera_right2 = self.world.spawn_actor(
                self.dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov) * 2)), attach_to=self.vehicle)

            for one_camera_idx, one_dvs_camera in enumerate([
                self.dvs_camera_left2, self.dvs_camera_left1,
                self.dvs_camera_mid,
                self.dvs_camera_right1, self.dvs_camera_right2
            ]):
                #             print("listen dvs:", one_camera_idx)
                one_dvs_camera.listen(lambda data, one_camera_idx=one_camera_idx: __get_dvs_data__(data, one_camera_idx))

            self.sensor_actors.append(self.dvs_camera_left2)
            self.sensor_actors.append(self.dvs_camera_left1)
            self.sensor_actors.append(self.dvs_camera_mid)
            self.sensor_actors.append(self.dvs_camera_right1)
            self.sensor_actors.append(self.dvs_camera_right2)

        #         print("\t dvs sensors init done.")
        if self.perception_type.__contains__("vidar"):

            def __get_vidar_data__(data, one_camera_idx):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                RGB = array[:, :, ::-1]
                Y = 0.2990 * RGB[:, :, 0] + 0.5870 * RGB[:, :, 1] + 0.1140 * RGB[:, :, 2]

                # print(self.vidar_data['img'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size, 1].shape)
                # print("!!!:", (self.vidar_data['spike'] * 255).astype(np.uint8).shape)

                self.vidar_data['frame'][one_camera_idx] = data.frame
                self.vidar_data['timestamp'][one_camera_idx] = data.timestamp
                self.vidar_data['voltage'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size] += Y.astype(np.uint8)

                # if one_camera_idx not in self.vidar_data['camera_done_idx']:
                #     self.vidar_data['camera_done_idx'].append(one_camera_idx)

                # if len(self.vidar_data['camera_done_idx']) == 5:
                #     self.vidar_data['camera_done_idx'] = []
                self.vidar_data['spike'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size] = \
                    self.vidar_data['voltage'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size] > 255
                self.vidar_data['voltage'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size] -= \
                    (self.vidar_data['spike'] * 255)[:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size].astype(np.uint8)

                self.vidar_data['img'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size, 1] = \
                    (self.vidar_data['spike'] * 255)[:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size].astype(np.uint8)

                # if len(self.vidar_data["captured_cam"]) != 5:
                #     self.vidar_data["captured_cam"].append(one_camera_idx)
                # else:
                #     self.vidar_data['captured_cam'].clear()
                #
                #     # rec
                #     if self.perception_type.__contains__("vidar") and self.perception_type.__contains__("rec"):
                #         from run_vidar_rec import run_vidar_rec
                #         out = run_vidar_rec(
                #             # x, y, p, t -> t, x, y, p
                #             self.dvs_data['events'][:, [3, 0, 1, 2]],
                #             self.rec_model, self.device, self.dvs_rec_args)
                #         # print(out.shape, out.dtype)
                #         # print(out[:5, :5])
                #         self.dvs_data['rec-img'][:, :, 0] = out

            self.vidar_camera_left2 = self.world.spawn_actor(
                self.vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov) * 2)),
                attach_to=self.vehicle)
            self.vidar_camera_left1 = self.world.spawn_actor(
                self.vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov) * 1)),
                attach_to=self.vehicle)
            self.vidar_camera_mid = self.world.spawn_actor(
                self.vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
            self.vidar_camera_right1 = self.world.spawn_actor(
                self.vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov) * 1)),
                attach_to=self.vehicle)
            self.vidar_camera_right2 = self.world.spawn_actor(
                self.vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov) * 2)),
                attach_to=self.vehicle)

            for one_camera_idx, one_vidar_camera in enumerate([
                self.vidar_camera_left2, self.vidar_camera_left1,
                self.vidar_camera_mid,
                self.vidar_camera_right1, self.vidar_camera_right2
            ]):
                #             print("listen dvs:", one_camera_idx)
                one_vidar_camera.listen(lambda data, one_camera_idx=one_camera_idx: __get_vidar_data__(data, one_camera_idx))

            self.sensor_actors.append(self.vidar_camera_left2)
            self.sensor_actors.append(self.vidar_camera_left1)
            self.sensor_actors.append(self.vidar_camera_mid)
            self.sensor_actors.append(self.vidar_camera_right1)
            self.sensor_actors.append(self.vidar_camera_right2)


        # Collision Sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self._collision_intensities_during_last_time_step = []

        self.sensor_actors.append(self.collision_sensor)
        #         print("set collision done")

        # print("\t collision sensor init done.")
        self.world.tick()

    def reset_weather(self):
        assert self.selected_weather in self.weather_params.keys()

        weather_params = self.weather_params[self.selected_weather]

        self.weather = self.world.get_weather()

        self.weather.cloudiness = weather_params["cloudiness"]
        self.weather.precipitation = weather_params["precipitation"]
        self.weather.precipitation_deposits = weather_params["precipitation_deposits"]
        self.weather.wind_intensity = weather_params["wind_intensity"]
        self.weather.fog_density = weather_params["fog_density"]
        self.weather.fog_distance = weather_params["fog_distance"]
        self.weather.wetness = weather_params["wetness"]
        self.weather.sun_azimuth_angle = weather_params["sun_azimuth_angle"]
        self.weather.sun_altitude_angle = weather_params["sun_altitude_angle"]

        self.world.set_weather(self.weather) 
        # or self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def step(self, action):
        rewards = []
        next_obs, done, info = None, None, None

        for _ in range(self.frame_skip):  # default 1
            # next_obs, reward, done, info = self._simulator_step(action, self.delta_seconds)
            next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info  # just last info?

    def get_reward(self):
        reward = sum(self.reward)
        self.reward = []
        return reward

    def get_perception(self):

        perception_data = self.perception_data

        # 根据perception_type进行预处理
        if self.perception_type.__contains__("rgb"):
            perception_data = np.transpose(perception_data, (2, 0, 1))
            perception_data = (perception_data / 255.).astype(np.float32)
            # print("perception_data:", perception_data.min(), perception_data.max())


        elif self.perception_type.__contains__("dvs"):
            # print("get self.perception_data:", len(self.perception_data))

            # 把events汇总
            events_window = np.concatenate(perception_data, axis=0)
            events_window = events_window[np.argsort(events_window[:, -1])]
            # print("events_window:", events_window.shape)
            # print(events_window)

            perception_data = events_window

            if self.perception_type.__contains__("stream"):
                pass
            elif self.perception_type.__contains__("rec"):
                # print("in dvs rec")

                from run_dvs_rec import run_dvs_rec
                perception_data = run_dvs_rec(
                    # x, y, p, t -> t, x, y, p
                    perception_data[:, [3, 0, 1, 2]],
                    self.rec_model, self.device, self.dvs_rec_args)
                # print(out.shape, out.dtype)
                # print(out[:5, :5])
                # print("raw dvs.shape:", perception_data.shape)
                perception_data = np.transpose(perception_data[..., np.newaxis], (2, 0, 1))
                perception_data = (perception_data / 255.).astype(np.float32)
                # print("perception_data:", perception_data.min(), perception_data.max())

                # print("after dvs rec.shape:", perception_data.shape)

        elif self.perception_type.__contains__("vidar"):
            if self.perception_type.__contains__("stream"):
                pass
            elif self.perception_type.__contains__("rec"):
                # print("in vidar rec")

                from run_vidar_rec import run_vidar_rec
                perception_data = run_vidar_rec(perception_data)
                # print("r", perception_data.shape, perception_data.dtype)
                # print(perception_data)
                perception_data = np.transpose(perception_data[..., np.newaxis], (2, 0, 1))
                # print("vidar rec:", perception_data.min(), perception_data.max())
                # print(perception_data[np.where((perception_data>0) & (perception_data<1))])
                # print("a", perception_data.shape, perception_data.dtype)

        # 重置
        # if self.perception_type.__contains__("rgb"):
        #     pass
        # else:
        #     self.perception_data.clear()
        self.perception_data = []

        return perception_data

    def _simulator_step(self, action, dt=0.1):

        if action is None and self.last_action is not None:
            action = self.last_action

        if action is not None:
            steer = float(action[0])
            throttle_brake = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

            self.last_action = action

        else:
            throttle, steer, brake = 0., 0., 0.

        assert 0.0 <= throttle <= 1.0
        assert -1.0 <= steer <= 1.0
        assert 0.0 <= brake <= 1.0
        vehicle_control = carla.VehicleControl(
            throttle=throttle,  # [0.0, 1.0]
            steer=steer,  # [-1.0, 1.0]
            brake=brake,  # [0.0, 1.0]
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(vehicle_control)


        self._control_all_walkers()

        # Advance the simulation and wait for the data.
        #         self.dvs_data["events"] = None
        self.frame = self.world.tick()


        # if self.spectator is not None:
        #     self.spectator.set_transform(
        #         carla.Transform(self.ego_vehicle.get_transform().location + carla.Location(z=40),
        #         carla.Rotation(pitch=-90)))

        info = {}
        info['reason_episode_ended'] = ''
        dist_from_center, vel_s, speed, done = self._dist_from_center_lane(self.vehicle, info)
        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step
        assert collision_intensities_during_last_time_step >= 0.
        colliding = float(collision_intensities_during_last_time_step > 0.)

        if colliding:
            self.collide_count += 1
        else:
            self.collide_count = 0
        if self.collide_count >= 20:
            print("Episode fail: too many collisions ({})! (frame {})".format(speed, self.collide_count))
            done = True

        #         reward = vel_s * dt / (1. + dist_from_center) - 1.0 * colliding - 0.1 * brake - 0.1 * abs(steer)

        collision_cost = 0.001 * collision_intensities_during_last_time_step
        reward = vel_s * dt - collision_cost - abs(steer)
        self.reward.append(reward)

        self.dist_s += vel_s * self.delta_seconds
        self.return_ += reward

        self.time_step += 1

        next_obs = {
            'video_frame': self.video_data['img'],
            'rgb_frame': self.rgb_data['img'],
        }

        if self.perception_type.__contains__("dvs"):
            next_obs.update({
                'dvs_frame': self.dvs_data['img'],
                'dvs_events': self.dvs_data['events']
            })
            self.perception_data.append(self.dvs_data['events'].copy())

            # if self.perception_type.__contains__("rec"):
            #     next_obs.update({
            #         'dvs_rec_img': self.dvs_data['rec-img'],
            #         'perception': self.dvs_data['rec-img']
            #     })
            #     self.perception_data.append(self.dvs_data['rec-img'])
            #
            # elif self.perception_type.__contains__("stream"):
            #     next_obs.update({
            #         'perception': self.dvs_data['events']
            #     })
            #     self.perception_data.append(self.dvs_data['events'])

        elif self.perception_type.__contains__("vidar"):
            next_obs.update({
                'vidar_frame': self.vidar_data['img']
            })
            self.perception_data.append(self.vidar_data['spike'].copy())

        elif self.perception_type.__contains__("rgb"):
            # next_obs.update({
            #     'perception': self.rgb_data['img']
            # })
            self.perception_data = self.rgb_data['img'].copy()

        info['crash_intensity'] = collision_intensities_during_last_time_step
        info['throttle'] = throttle
        info['steer'] = steer
        info['brake'] = brake
        info['distance'] = vel_s * dt
        """
        next_obs['dvs_frame'].shape: (84, 420, 3)
        self.observation_space.shape: (2, 84, 420)
        """

        # debugging - to inspect images:
        #         import matplotlib.pyplot as plt
        #         import pdb; pdb.set_trace()
        #         plt.imshow(next_obs['dvs_frame'])
        #         plt.show()
        #         next_obs["dvs_frame"] = np.transpose(next_obs["dvs_frame"], [2, 0, 1])
        #         print("self.observation_space.shape:", self.observation_space.shape)
        #         print('next_obs["dvs_frame"][::2,:,:].shape:', next_obs["dvs_frame"][::2,:,:].shape)

        #         assert next_obs["dvs_frame"][::2,:,:].shape == self.observation_space.shape  # (2, 84, 420)
        #         assert next_obs["dvs_events"].shape[1] == self.observation_space.shape[1]
        if self.time_step >= self.max_episode_steps:
            info['reason_episode_ended'] = 'success'
            print("Episode success: I've reached the episode horizon ({}).".format(self.max_episode_steps))
            done = True
        #         if speed < 0.02 and self.time_step >= 8 * (self.fps) and self.time_step % 8 * (self.fps) == 0:  # a hack, instead of a counter
        if speed < 0.02 and self.time_step >= (2 * self.max_fps) and self.time_step % (2 * self.max_fps) == 0:  # a hack, instead of a counter
            print("Episode fail: speed too small ({}), think I'm stuck! (frame {})".format(speed, self.time_step))
            info['reason_episode_ended'] = 'stuck'
            done = True

        return next_obs, reward, done, info

    def finish(self):
        print('destroying actors.')
        actor_list = self.world.get_actors()
        for one_actor in actor_list:
            one_actor.destroy()
        time.sleep(0.5)
        print('done.')

