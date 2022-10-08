# !/usr/bin/python3
# -*- coding: utf-8 -*-
# CARLA 0.9.13 environment

import os
import imageio
import numpy as np
from PIL import ImageFont, ImageDraw, Image


class VideoRecorder(object):
    def __init__(self, dir_name, min_fps, max_fps, control_hz=None):
        self.dir_name = dir_name
        self.min_fps = min_fps
        self.max_fps = max_fps
        if control_hz:
            self.control_hz = max_fps
        else:
            self.control_hz = control_hz

        self.video_frames = []
        self.rgb_frames = []
        self.dvs_frames = []
        self.perception_frames = []
        self.vidar_frames = []
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


    def init(self, enabled=True):

        self.video_frames = []
        self.rgb_frames = []
        self.dvs_frames = []
        self.vidar_frames = []
        self.perception_frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, obs, perception, vehicle=None):

        if self.enabled:
            if "video_frame" in obs.keys():

                video_frame = obs["video_frame"].copy()

                if vehicle is not None:
                    height, width = video_frame.shape[0:2]  # 600, 800
                    
                    video_frame = Image.fromarray(video_frame)#.convert("P")
                    draw = ImageDraw.Draw(video_frame)
                    # print(video_frame.mode)

                    control = vehicle.get_control()
                    velocity = vehicle.get_velocity()

                    dw = width - 150
                    dh = 20
                    draw.text((dw, dh), f"throttle: {control.throttle:.5f}", fill = (255, 255, 255))
                    draw.text((dw, dh+20), f"steer: {control.steer:.5f}", fill = (255, 255, 255))
                    draw.text((dw, dh+40), f"brake: {control.brake:.5f}", fill = (255, 255, 255))
                    draw.text((dw, dh+60), f"vx: {velocity.x:.5f}", fill = (255, 255, 255))
                    draw.text((dw, dh+80), f"vy: {velocity.y:.5f}", fill = (255, 255, 255))

                    draw.text((dw, dh+120), f"MAX FPS: {self.max_fps}", fill = (255, 255, 255))
                    draw.text((dw, dh+140), f"MIN FPS: {self.min_fps}", fill = (255, 255, 255))
                    draw.text((dw, dh+160), f"Control Hz: {self.control_hz}", fill = (255, 255, 255))
                    # video_frame.show()
                    video_frame = np.array(video_frame)

                self.video_frames.append(video_frame)

            self.perception_frames.append(
                (np.transpose(perception, (1, 2, 0)) * 255.).astype(np.uint8)
            )

            if "rgb_frame" in obs.keys():    
                self.rgb_frames.append(obs["rgb_frame"].copy())
            if "dvs_frame" in obs.keys():
                self.dvs_frames.append(obs["dvs_frame"].copy())
            if "vidar_frame" in obs.keys():
                self.vidar_frames.append(obs["vidar_frame"].copy())

    def save(self, file_name, type="mp4"):
        if self.enabled:
            video_frames_path = os.path.join(self.dir_name, file_name + f"-video.{type}")
            rgb_frames_path = os.path.join(self.dir_name, file_name + f"-rgb.{type}")
            dvs_frames_path = os.path.join(self.dir_name, file_name + f"-dvs.{type}")
            vidar_frames_path = os.path.join(self.dir_name, file_name + f"-vidar.{type}")
            perception_frames_path = os.path.join(self.dir_name, file_name + f"-perception.{type}")

            if len(self.perception_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(perception_frames_path, self.perception_frames, fps=self.control_hz, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(perception_frames_path, self.perception_frames, duration=1/self.control_hz)

            if len(self.video_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(video_frames_path, self.video_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(video_frames_path, self.video_frames, duration=1/self.max_fps)

            if len(self.rgb_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(rgb_frames_path, self.rgb_frames, fps=self.min_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(rgb_frames_path, self.rgb_frames, duration=1/self.min_fps)
            if len(self.dvs_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(dvs_frames_path, self.dvs_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(dvs_frames_path, self.dvs_frames, duration=1/self.max_fps)
            if len(self.vidar_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(vidar_frames_path, self.vidar_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(vidar_frames_path, self.vidar_frames, duration=1/self.max_fps)   
            