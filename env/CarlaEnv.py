# !/usr/bin/python3
# -*- coding: utf-8 -*-
# CARLA 0.9.13 environment

"""
sudo docker run --privileged --user carla --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla-add:0.9.13 /bin/bash ./CarlaUE4.sh -world-port=12321 -RenderOffScreen
"""

import carla

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
                 perception_type, num_cameras, rl_image_size, fov, max_fps, min_fps,
                 max_episode_steps, frame_skip,
                 ):

        self.frame_skip = frame_skip

        self.carla_rpc_port = carla_rpc_port
        self.carla_tm_port = carla_tm_port
        self.carla_timeout = carla_timeout
        self.weather_params = weather_params
        self.scenario_params = scenario_params

        # rgb-frame, dvs-frame, dvs-stream, dvs-vidar-stream
        self.perception_type = perception_type  # ↑↑↑↑↑↑↑↑↑
        self.num_cameras = num_cameras
        self.rl_image_size = rl_image_size
        self.fov = fov
        self.max_fps = max_fps
        self.min_fps = min_fps
        self.max_episode_steps = max_episode_steps

        self.selected_weather = selected_weather
        self.selected_scenario = selected_scenario

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

        # reset
        self.reset()

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
            self.observation_space.shape = (2, self.rl_image_size, self.num_cameras * self.rl_image_size)
            self.observation_space.dtype = np.dtype(np.uint8)
        elif self.perception_type.__contains__("stream"):
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
            print("Episode fail: goal waypoint is off the road! (frame %d)" % self.count)
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
                print("Episode done: no more waypoints left. (frame %d)" % self.count)
                info['reason_episode_ended'] = 'no_waypoints'
                done, vel_s = True, 0.
            else:
                location_ahead = next_goal_waypoint[0].transform.location
                highway_vector = np.array([location_ahead.x, location_ahead.y]) - goal_xy
                highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)
                done = False

        # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        #         if vehicle_velocity.z > 1. and self.count < 20:
        #             print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_velocity.z, self.count))
        #             done = True
        #         if vehicle_location.z > 0.5 and self.count < 20:
        #             print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_location.z, self.count))
        #             done = True

        return dist, vel_s, speed, done

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        print('Collision (intensity {})'.format(intensity))
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

    def _clear_all_objects(self):
        # remove all vehicles, walkers, and sensors (in case they survived)
        # self.world.tick()

        # actor_list = self.world.get_actors()

        for one_sensor_actor in self.sensor_actors:
            one_sensor_actor.destroy()

        # self.vidar_data['voltage'] = np.zeros((self.obs_size, self.obs_size), dtype=np.uint16)

        for one_vehicle_actor in self.vehicle_actors:
            if one_vehicle_actor.is_alive:
                one_vehicle_actor.destroy()

        for one_walker_ai_actor in self.walker_ai_actors:
            if one_walker_ai_actor.is_alive:
                one_walker_ai_actor.stop()
                one_walker_ai_actor.destroy()

        for one_walker_actor in self.walker_actors:
            if one_walker_actor.is_alive:
                one_walker_actor.destroy()

        # for actor_filter in ['vehicle.*', 'controller.ai.walker', 'walker.*']:
        #     for actor in self.world.get_actors().filter(actor_filter):
        #         if actor.is_alive:
        #             if actor.type_id == 'controller.ai.walker':
        #                 actor.stop()
        #             actor.destroy()

        # for one_vehicle_actor in self.vehicle_actors:
        #     one_vehicle_actor.destroy()

        # for one_sensor_actor in self.sensor_actors:
        #     one_sensor_actor.stop() 
        #     one_sensor_actor.destroy()

        # for one_walker_ai_actor in self.walker_ai_actors:
        #     one_walker_ai_actor.destroy()

        # for one_walker_actor in self.walker_actors:
        #     one_walker_actor.destroy()

        # for sensor in actor_list.filter("*sensor*"):
        #     sensor.destroy()

        # for vehicle in actor_list.filter("*vehicle*"):
        #     vehicle.destroy()

        # for walker in actor_list.filter("*walker*"):
        #     walker.destroy()

        self.vehicle_actors = []
        self.sensor_actors = []
        self.walker_actors = []
        self.walker_ai_actors = []

        self.world.tick()

    def reset(self):

        # remove event callbacks
        #         self.world.remove_on_tick(self.sync_mode.callback_id)
        # [one_sensor.stop() for one_sensor in self.sensors]

        #         self.world = self.client.reload_world() # 0.9.9.4
        #         self.world = self.client.reload_world(reset_settings=False)

        self._clear_all_objects()

        self.client.reload_world(reset_settings=True)
        self.world = self.client.load_world(self.scenario_params[self.selected_scenario]["map"])
        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # tm
        self.tm = self.client.get_trafficmanager(self.carla_tm_port)
        self.tm_port = self.tm.get_port()
        self.tm.set_global_distance_to_leading_vehicle(2.0)
        
        # lm
        self.lm = self.world.get_lightmanager()

        # reset
        self.reset_sync_mode(False)

        self.reset_surrounding_vehicles()
        self.reset_special_vehicles()
        self.reset_walkers()
        self.reset_ego_vehicle()

        self.reset_weather()
        self.reset_sensors()

        self.reset_sync_mode(True)

        self.count = 0
        self.dist_s = 0
        self.return_ = 0
        self.velocities = []

        # warm up !!!!!!
        # take some steps to get ready for the dvs camera, walkers, and vehicles
        warm_up_max_steps = 5
        self.vehicle.set_autopilot(True, self.carla_tm_port)
        while abs(self.vehicle.get_velocity().x) < 0.02:
            #             print("!!!take one init step", warm_up_max_steps, self.vehicle.get_control(), self.vehicle.get_velocity())
            self.world.tick()
            #             action = self.compute_steer_action()
            #             obs, _, _, _ = self.step(action=action)
            #             self.count -= 1
            warm_up_max_steps -= 1
            if warm_up_max_steps < 0 and self.dvs_data['events'] is not None:
                break
        self.vehicle.set_autopilot(False, self.carla_tm_port)

        obs, _, _, _ = self.step(None)

        print("carla env reset done.")

        return obs

    def reset_sync_mode(self, synchronous_mode=True):
        self.delta_seconds = 1.0 / self.max_fps

        #         self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=synchronous_mode,
            fixed_delta_seconds=self.delta_seconds))
        self.tm.set_synchronous_mode(synchronous_mode)

    def reset_surrounding_vehicles(self):
        total_surrounding_veh_num = 0
        veh_bp = self.bp_lib.filter('vehicle.*')
        veh_bp = [x for x in veh_bp if int(x.get_attribute('number_of_wheels')) == 4]

        for one_type in ["same_dir_veh", "oppo_dir_veh"]:
            # for one_type in ["same_dir_veh"]:

            one_type_params = self.scenario_params[self.selected_scenario][one_type]

            print("now at:", one_type)

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
                        self.tm.auto_lane_change(vehicle, True)
                        self.tm.vehicle_percentage_speed_difference(
                            vehicle, one_type_params[one_part]["speed"])
                        self.tm.ignore_lights_percentage(vehicle, 100)
                        self.tm.ignore_signs_percentage(vehicle, 100)
                        self.world.tick()

                        self.vehicle_actors.append(vehicle)

                        veh_num -= 1
                        total_surrounding_veh_num += 1
                        print(f"\t spawn vehicle: {total_surrounding_veh_num}, at {veh_pos.location}")


    def reset_special_vehicles(self):
        pass

    def reset_walkers(self):
        walker_bp = self.world.get_blueprint_library().filter('walker.*')
        total_surrounding_walker_num = 0

        walker_params = self.scenario_params[self.selected_scenario]["walker"]

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
                    walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                    walker_controller_actor = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker_actor)
                    # start walker
                    walker_controller_actor.start()
                    # set walk to random point
                    #             walker_controller_actor.go_to_location(world.get_random_location_from_navigation())
                    rand_destination = carla.Location(
                        x=np.random.uniform(walker_params[one_part]["dest"]["x"][0], walker_params[one_part]["dest"]["x"][1]),
                        y=random.choice([walker_params[one_part]["dest"]["y"][0], walker_params[one_part]["dest"]["y"][1]]),
                        z=0.
                    )
                    walker_controller_actor.go_to_location(rand_destination)
                    # random max speed (default is 1.4 m/s)
                    walker_controller_actor.set_max_speed(
                        np.random.uniform(
                            walker_params[one_part]["speed"][0],
                            walker_params[one_part]["speed"][1]
                        ))

                    self.walker_actors.append(walker_actor)
                    self.walker_ai_actors.append(walker_controller_actor)

                    self.world.tick()
                    walker_num -= 1
                    total_surrounding_walker_num += 1
                    print(f"\t spawn walker: {total_surrounding_walker_num}, at {walker_pos.location}")

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
        max_ego_spawn_times = 20
        while True:


            if ego_spawn_times > max_ego_spawn_times:
                #                 print("\tspawn ego vehicle times > max_ego_spawn_times")
                self.reset()

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
                if dis > 6:
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

                #                 physics_control = self.vehicle.get_physics_control()
                #                 physics_control.gear_switch_time=0.01
                #                 physics_control.damping_rate_zero_throttle_clutch_engaged=physics_control.damping_rate_zero_throttle_clutch_disengaged
                #                 self.vehicle.apply_physics_control(physics_control)
                #                 self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1, manual_gear_shift=True, gear=1))

                # velocity
                self.tm.vehicle_percentage_speed_difference(self.vehicle, ego_veh_params["speed"])

                self.vehicle_actors.append(self.vehicle)

                # AUTO pilot
                #                 vehicle.set_autopilot(True, tm_port)
                #                 tm.ignore_lights_percentage(vehicle, 100)
                #                 tm.ignore_signs_percentage(vehicle, 100)

                #                 self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                #                 self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                # 0.9.9.4
                #                 self.vehicle.set_velocity(carla.Vector3D(0, 0, 0))
                #                 self.vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
                # self.vehicle.set_light_state(carla.libcarla.VehicleLightState.HighBeam)  # HighBeam # LowBeam  # All
                break
            else:

                ego_spawn_times += 1

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
                             'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8)}
        if self.perception_type.__contains__("vidar"):
            self.vidar_data = {'frame': [0] * self.num_cameras, 'timestamp': [0.0] * self.num_cameras,
                               'voltage': np.zeros((self.rl_image_size, self.rl_image_size*self.num_cameras), dtype=np.uint16),
                               'spike': np.zeros((self.rl_image_size, self.rl_image_size*self.num_cameras), dtype=np.bool_),
                               'img': np.zeros((self.rl_image_size, self.rl_image_size*self.num_cameras, 3), dtype=np.uint8)}

        self.frame = None

        #         def on_tick_func(data):
        #             self.dvs_data["events"] = None
        #         self.world.on_tick(on_tick_func)

        # Third person perspective
        def get_video_data(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.video_data['frame'] = data.frame
            self.video_data['timestamp'] = data.timestamp
            self.video_data['img'] = array

        self.third_person_camera_rgb = self.world.spawn_actor(
            self.bp_lib.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=3.5), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.third_person_camera_rgb.listen(lambda data: get_video_data(data))
        # self.sensors.append(self.third_person_camera_rgb)

        #         print("\t video sensor init done.")

        # we'll use up to five cameras, which we'll stitch together
        location = carla.Location(x=1.6, z=1.7)

        # Perception RGB sensor
        def get_rgb_data(data, one_camera_idx):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.rgb_data['frame'][one_camera_idx] = data.frame
            self.rgb_data['timestamp'][one_camera_idx] = data.timestamp
            self.rgb_data['img'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size, :] = array

        rgb_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('sensor_tick', f'{1 / self.min_fps}')
        rgb_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        rgb_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        rgb_camera_bp.set_attribute('fov', str(self.fov))
        rgb_camera_bp.set_attribute('enable_postprocess_effects', str(True))

        self.rgb_camera_left2 = self.world.spawn_actor(
            rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=-2 * float(self.fov))),
            attach_to=self.vehicle)
        self.rgb_camera_left1 = self.world.spawn_actor(
            rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov))),
            attach_to=self.vehicle)
        self.rgb_camera_mid = self.world.spawn_actor(
            rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
            attach_to=self.vehicle)
        self.rgb_camera_right1 = self.world.spawn_actor(
            rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov))),
            attach_to=self.vehicle)
        self.rgb_camera_right2 = self.world.spawn_actor(
            rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=2 * float(self.fov))),
            attach_to=self.vehicle)

        for one_camera_idx, one_rgb_camera in enumerate([
            self.rgb_camera_left2, self.rgb_camera_left1,
            self.rgb_camera_mid,
            self.rgb_camera_right1, self.rgb_camera_right2
        ]):
            #             print("listen rgb:", one_camera_idx)
            one_rgb_camera.listen(lambda data, one_camera_idx=one_camera_idx: get_rgb_data(data, one_camera_idx))

        self.sensor_actors.append(self.rgb_camera_left2)
        self.sensor_actors.append(self.rgb_camera_left1)
        self.sensor_actors.append(self.rgb_camera_mid)
        self.sensor_actors.append(self.rgb_camera_right1)
        self.sensor_actors.append(self.rgb_camera_right2)

        #         print("\t rgb sensors init done.")

        # Perception DVS sensor
        if self.perception_type.__contains__("dvs"):
            def get_dvs_data(data, one_camera_idx):
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


            dvs_camera_bp = self.bp_lib.find('sensor.camera.dvs')
            dvs_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
            #         dvs_camera_bp.set_attribute('positive_threshold', str(0.3))
            #         dvs_camera_bp.set_attribute('negative_threshold', str(0.3))
            #         dvs_camera_bp.set_attribute('sigma_positive_threshold', str(0))
            #         dvs_camera_bp.set_attribute('sigma_negative_threshold', str(0))
            dvs_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
            dvs_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
            dvs_camera_bp.set_attribute('fov', str(self.fov))
            dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))

            self.dvs_camera_left2 = self.world.spawn_actor(
                dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov) * 2)), attach_to=self.vehicle)
            self.dvs_camera_left1 = self.world.spawn_actor(
                dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov) * 1)), attach_to=self.vehicle)
            self.dvs_camera_mid = self.world.spawn_actor(
                dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
            self.dvs_camera_right1 = self.world.spawn_actor(
                dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov) * 1)), attach_to=self.vehicle)
            self.dvs_camera_right2 = self.world.spawn_actor(
                dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov) * 2)), attach_to=self.vehicle)

            for one_camera_idx, one_dvs_camera in enumerate([
                self.dvs_camera_left2, self.dvs_camera_left1,
                self.dvs_camera_mid,
                self.dvs_camera_right1, self.dvs_camera_right2
            ]):
                #             print("listen dvs:", one_camera_idx)
                one_dvs_camera.listen(lambda data, one_camera_idx=one_camera_idx: get_dvs_data(data, one_camera_idx))

            self.sensor_actors.append(self.dvs_camera_left2)
            self.sensor_actors.append(self.dvs_camera_left1)
            self.sensor_actors.append(self.dvs_camera_mid)
            self.sensor_actors.append(self.dvs_camera_right1)
            self.sensor_actors.append(self.dvs_camera_right2)

        #         print("\t dvs sensors init done.")
        if self.perception_type.__contains__("vidar"):

            def get_vidar_data(data, one_camera_idx):
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

            vidar_camera_bp = self.bp_lib.find('sensor.camera.rgb')
            vidar_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
            vidar_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
            vidar_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
            vidar_camera_bp.set_attribute('fov', str(self.fov))
            vidar_camera_bp.set_attribute('enable_postprocess_effects', str(True))

            self.vidar_camera_left2 = self.world.spawn_actor(
                vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov) * 2)),
                attach_to=self.vehicle)
            self.vidar_camera_left1 = self.world.spawn_actor(
                vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov) * 1)),
                attach_to=self.vehicle)
            self.vidar_camera_mid = self.world.spawn_actor(
                vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
            self.vidar_camera_right1 = self.world.spawn_actor(
                vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov) * 1)),
                attach_to=self.vehicle)
            self.vidar_camera_right2 = self.world.spawn_actor(
                vidar_camera_bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov) * 2)),
                attach_to=self.vehicle)

            for one_camera_idx, one_vidar_camera in enumerate([
                self.vidar_camera_left2, self.vidar_camera_left1,
                self.vidar_camera_mid,
                self.vidar_camera_right1, self.vidar_camera_right2
            ]):
                #             print("listen dvs:", one_camera_idx)
                one_vidar_camera.listen(lambda data, one_camera_idx=one_camera_idx: get_vidar_data(data, one_camera_idx))

            self.sensor_actors.append(self.vidar_camera_left2)
            self.sensor_actors.append(self.vidar_camera_left1)
            self.sensor_actors.append(self.vidar_camera_mid)
            self.sensor_actors.append(self.vidar_camera_right1)
            self.sensor_actors.append(self.vidar_camera_right2)


        # Collision Sensor
        bp = self.bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
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

    def step(self, action):
        rewards = []
        next_obs, done, info = None, None, None

        for _ in range(self.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(action, self.delta_seconds)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info  # just last info?

    def _simulator_step(self, action, dt):

        if action is not None:
            steer = float(action[0])
            throttle_brake = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake
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

        # Advance the simulation and wait for the data.
        #         self.dvs_data["events"] = None
        self.frame = self.world.tick()

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

        collision_cost = 0.0001 * collision_intensities_during_last_time_step
        reward = vel_s * dt - collision_cost - abs(steer)

        self.dist_s += vel_s * dt
        self.return_ += reward

        self.count += 1

        next_obs = {
            'video_frame': self.video_data['img'],
            'rgb_frame': self.rgb_data['img'],
        }

        if self.perception_type.__contains__("dvs"):
            next_obs.update({
                'dvs_frame': self.dvs_data['img'],
                'dvs_events': self.dvs_data['events']
            })
        if self.perception_type.__contains__("vidar"):
            next_obs.update({
                'vidar_frame': self.vidar_data['img']
            })

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
        if self.count >= self.max_episode_steps:
            info['reason_episode_ended'] = 'success'
            print("Episode success: I've reached the episode horizon ({}).".format(self.max_episode_steps))
            done = True
        #         if speed < 0.02 and self.count >= 8 * (self.fps) and self.count % 8 * (self.fps) == 0:  # a hack, instead of a counter
        if speed < 0.02 and self.count >= 100 and self.count % 100 == 0:  # a hack, instead of a counter
            print("Episode fail: speed too small ({}), think I'm stuck! (frame {})".format(speed, self.count))
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

if __name__ == '__main__':
    import json

    # read config files
    with open('./cfg/weather.json', 'r', encoding='utf8') as fff:
        weather_params = json.load(fff)
    with open('./cfg/scenario.json', 'r', encoding='utf8') as fff:
        scenario_params = json.load(fff)

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
