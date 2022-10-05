import torch
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
import time

from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options



def run_dvs_rec(event_window, args):

    """
    path_to_model = "./tools/rpg_e2vid/pretrained/E2VID_lightweight.path.tar"
    use_gpu = True
    fixed_duration = True
    window_duration (milliseconds) = 33.33
    skipevents = 0
    suboffset = 0
    display = True
    show_events = True
    event_display_mode = red-blue
    num_events_per_pixel = 0.35
    num_bins_to_show = -1
    display_border_crop = 0
    display_wait_time = 1
    compute_voxel_grid_on_cpu = False
    output_folder = None
    dataset_name = "reconstruction"

    hot_pixels_file = False
    unsharp_mask_amount = 0.3
    unsharp_mask_sigma = 1.0
    bilateral_filter_sigma = 0.0
    flip = False
    Imin = 0.0
    Imax = 1.0
    auto_hdr = False
    auto_hdr_median_filter_size = 10
    color = False
    no_normalize = False
    no_recurrent = False
    """

    # Load model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    """ Read chunks of events using Pandas """

    # reconstruct image
    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                num_bins=model.num_bins,  # 5
                                                width=args.width,
                                                height=args.height,
                                                device=device)

    num_events_in_window = event_window.shape[0]
    reconstructor.update_reconstruction(event_tensor)

