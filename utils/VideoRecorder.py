# !/usr/bin/python3
# -*- coding: utf-8 -*-
# CARLA 0.9.13 environment

import os
import imageio
import numpy as np
from PIL import ImageFont, ImageDraw, Image


class VideoRecorder(object):
    def __init__(self, dir_name, min_fps, max_fps):
        self.dir_name = dir_name
        self.min_fps = min_fps
        self.max_fps = max_fps

        self.video_frames = []
        self.rgb_frames = []
        self.dvs_frames = []
        self.dvs_rec_frames = []
        self.vidar_frames = []
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


    def init(self, enabled=True):

        self.video_frames = []
        self.rgb_frames = []
        self.dvs_frames = []
        self.vidar_frames = []
        self.dvs_rec_frames = []
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
                    draw.text((dw, dh), f"throttle: {control.throttle:.5f}", fill = (255, 255, 255))
                    draw.text((dw, dh+20), f"steer: {control.steer:.5f}", fill = (255, 255, 255))
                    draw.text((dw, dh+40), f"brake: {control.brake:.5f}", fill = (255, 255, 255))
                    draw.text((dw, dh+60), f"vx: {velocity.x:.5f}", fill = (255, 255, 255))
                    draw.text((dw, dh+80), f"vy: {velocity.y:.5f}", fill = (255, 255, 255))
                    # video_frame.show()
                    video_frame = np.array(video_frame)

                self.video_frames.append(video_frame)


            if "rgb_frame" in obs.keys():    
                self.rgb_frames.append(obs["rgb_frame"].copy())
            if "dvs_frame" in obs.keys():
                self.dvs_frames.append(obs["dvs_frame"].copy())
            if "dvs_rec_img" in obs.keys():
                self.dvs_rec_frames.append(obs["dvs_rec_img"].copy())
            if "vidar_frame" in obs.keys():
                self.vidar_frames.append(obs["vidar_frame"].copy())

    def save(self, file_name, type="mp4"):
        if self.enabled:
            video_frames_path = os.path.join(self.dir_name, file_name + f"-video.{type}")
            rgb_frames_path = os.path.join(self.dir_name, file_name + f"-rgb.{type}")
            dvs_frames_path = os.path.join(self.dir_name, file_name + f"-dvs.{type}")
            dvs_rec_frames_path = os.path.join(self.dir_name, file_name + f"-dvs-rec.{type}")
            vidar_frames_path = os.path.join(self.dir_name, file_name + f"-vidar.{type}")
            
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
            if len(self.dvs_rec_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(dvs_rec_frames_path, self.dvs_rec_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(dvs_rec_frames_path, self.dvs_rec_frames, duration=1/self.max_fps)
            if len(self.vidar_frames) > 0:
                if type == "mp4":
                    imageio.mimsave(vidar_frames_path, self.vidar_frames, fps=self.max_fps, macro_block_size=2)
                elif type == "gif":
                    imageio.mimsave(vidar_frames_path, self.vidar_frames, duration=1/self.max_fps)   
            