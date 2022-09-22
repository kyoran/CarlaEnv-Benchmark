# !/usr/bin/python3
# -*- coding: utf-8 -*-
# CARLA 0.9.13 environment

import os
import imageio

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

    def record(self, obs):
        if self.enabled:
            if "video_frame" in obs.keys():
                self.video_frames.append(obs["video_frame"].copy())
            if "rgb_frame" in obs.keys():    
                self.rgb_frames.append(obs["rgb_frame"].copy())
            if "dvs_frame" in obs.keys():
                self.dvs_frames.append(obs["dvs_frame"].copy())
            if "vidar_frame" in obs.keys():
                self.vidar_frames.append(obs["vidar_frame"].copy())

    def save(self, file_name):
        if self.enabled:
            video_frames_path = os.path.join(self.dir_name, "video_frames_" + file_name)
            rgb_frames_path = os.path.join(self.dir_name, "rgb_frames_" + file_name)
            dvs_frames_path = os.path.join(self.dir_name, "dvs_frames_" + file_name)
            vidar_frames_path = os.path.join(self.dir_name, "vidar_frames_" + file_name)
            
            if len(self.video_frames) > 0:
                imageio.mimsave(video_frames_path, self.video_frames, fps=self.fps, macro_block_size=2)
            if len(self.rgb_frames) > 0:
                imageio.mimsave(rgb_frames_path, self.rgb_frames, fps=self.fps, macro_block_size=2)
            if len(self.dvs_frames) > 0:
                imageio.mimsave(dvs_frames_path, self.dvs_frames, fps=self.fps, macro_block_size=2)            
            if len(self.vidar_frames) > 0:
                imageio.mimsave(vidar_frames_path, self.vidar_frames, fps=self.fps, macro_block_size=2)    
            