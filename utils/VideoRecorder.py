# !/usr/bin/python3
# -*- coding: utf-8 -*-
# CARLA 0.9.13 environment

import os
import imageio
import numpy as np
from PIL import ImageFont, ImageDraw, Image


class VideoRecorder(object):
    def __init__(self, dir_name, fps):
        self.dir_name = dir_name
        self.fps = fps

        self.video_frames = []
        self.rgb_frames = []
        self.dvs_frames = []
        self.vidar_frames = []
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


    def init(self, enabled=True):

        self.video_frames = []
        self.rgb_frames = []
        self.dvs_frames = []
        self.vidar_frames = []
        
        self.enabled = self.dir_name is not None and enabled

    def record(self, obs, vehicle=None):
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
                    draw.text((dw, dh), f"throttle: {control.throttle:.2f}", fill = (255, 255, 255))
                    draw.text((dw, dh+20), f"steer: {control.steer:.2f}", fill = (255, 255, 255))
                    draw.text((dw, dh+40), f"brake: {control.brake:.2f}", fill = (255, 255, 255))
                    draw.text((dw, dh+60), f"vx: {velocity.x:.2f}", fill = (255, 255, 255))
                    draw.text((dw, dh+80), f"vy: {velocity.y:.2f}", fill = (255, 255, 255))
                    # video_frame.show()
                    video_frame = np.array(video_frame)

                self.video_frames.append(video_frame)


            if "rgb_frame" in obs.keys():    
                self.rgb_frames.append(obs["rgb_frame"].copy())
            if "dvs_frame" in obs.keys():
                self.dvs_frames.append(obs["dvs_frame"].copy())
            if "vidar_frame" in obs.keys():
                self.vidar_frames.append(obs["vidar_frame"].copy())

    def save(self, file_name):
        if self.enabled:
            video_frames_path = os.path.join(self.dir_name, file_name + "-video.mp4")
            rgb_frames_path = os.path.join(self.dir_name, file_name + "-rgb.mp4")
            dvs_frames_path = os.path.join(self.dir_name, file_name + "-dvs.mp4")
            vidar_frames_path = os.path.join(self.dir_name, file_name + "-vidar.mp4")
            
            if len(self.video_frames) > 0:
                imageio.mimsave(video_frames_path, self.video_frames, fps=self.fps, macro_block_size=2)
            if len(self.rgb_frames) > 0:
                imageio.mimsave(rgb_frames_path, self.rgb_frames, fps=self.fps, macro_block_size=2)
            if len(self.dvs_frames) > 0:
                imageio.mimsave(dvs_frames_path, self.dvs_frames, fps=self.fps, macro_block_size=2)            
            if len(self.vidar_frames) > 0:
                imageio.mimsave(vidar_frames_path, self.vidar_frames, fps=self.fps, macro_block_size=2)    
            