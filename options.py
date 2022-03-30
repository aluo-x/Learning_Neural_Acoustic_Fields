import argparse
import torch
import os
import random
import numpy as np

def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def list_float_flag(s):
    return [float(_) for _ in list(s)]

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        parser = self.parser
        parser.add_argument('--save_loc', default="./results", type=str)


        parser.add_argument('--apt', default='apartment_1', choices=['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2'], type=str)
        parser.add_argument('--exp_name', default="{}")

        # dataset arguments
        parser.add_argument('--coor_base', default="./metadata/replica", type=str)  # Location of the training index to coordinate mapping
        parser.add_argument('--spec_base', default="./metadata/magnitudes", type=str)  # Location of the actual training spectrograms
        parser.add_argument('--mean_std_base', default="./metadata/mean_std", type=str)  # Location of sound mean_std data
        parser.add_argument('--minmax_base', default="./metadata/minmax", type=str)  # Location of room bbox data
        parser.add_argument('--wav_base', default="/media/aluo/big2/soundspaces_full/binaural_rirs/replica", type=str)  # Location of impulses in raw .wav format
        parser.add_argument('--split_loc', default="./metadata/train_test_split/", type=str) # Where the train test split is stored


        # baseline arguments
        parser.add_argument('--opus_enc', default="/home/aluo/anaconda3/envs/torch110/bin/opusenc", type=str)  # Opus encoder path -- opusenc opus-tools 0.2 (using libopus 1.3.1)
        parser.add_argument('--opus_dec', default="/home/aluo/anaconda3/envs/torch110/bin/opusdec", type=str)  # Opus decoder path -- opusdec opus-tools 0.2 (using libopus 1.3.1)
        parser.add_argument('--ffmpeg', default="/home/aluo/Downloads/ffmpeg5/ffmpeg-5.0-amd64-static/ffmpeg", type=str)  # ffmpeg 5.0 path, used for AAC-LC -- ffmpeg version 5.0-static https://johnvansickle.com/ffmpeg/  Copyright (c) 2000-2022 the FFmpeg developers built with gcc 8 (Debian 8.3.0-6)
        parser.add_argument('--aac_write_path', default="/media/aluo/big2/soundspaces_full/aac_enc_test", type=str) # Where do we write the aac encoded-decoded data
        parser.add_argument('--opus_write_path', default="/media/aluo/big2/soundspaces_full/opus_enc_test", type=str) # Where do we write the opus encoded-decoded data
        parser.add_argument('--ramdisk_path', default="/mnt/ramdisk", type=str) # RAMdisk for acceleration

        # training arguments
        parser.add_argument('--gpus', default=4, type=int) # Number of GPUs to use
        parser.add_argument('--epochs', default=200, type=int) # Total epochs to train for
        parser.add_argument('--resume', default=0, type=bool_flag) # Load weights or not from latest checkpoint
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--reg_eps', default=1e-1, type=float) # Noise to regularize positions
        parser.add_argument('--pixel_count', default=2000, type=int)  # Noise to regularize positions
        parser.add_argument('--lr_init', default=5e-4, type=float)  # Starting learning rate
        parser.add_argument('--lr_decay', default=1e-1, type=float)  # Learning rate decay rate

        # network arguments
        parser.add_argument('--layers', default=8, type=int) # Number of layers in the network
        parser.add_argument('--grid_gap', default=0.25, type=float) # How far are the grid points spaced
        parser.add_argument('--bandwith_init', default=0.25, type=float) # Initial bandwidth of the grid
        parser.add_argument('--features', default=512, type=int) # Number of neurons in the network for each layer
        parser.add_argument('--grid_features', default=64, type=int) # Number of neurons in the grid
        parser.add_argument('--position_float', default=0.1, type=float) # Amount the position of each grid cell can float (up or down)
        parser.add_argument('--min_bandwidth', default=0.1, type=float) # Minimum bandwidth for clipping
        parser.add_argument('--max_bandwidth', default=0.5, type=float) # Maximum bandwidth for clipping
        parser.add_argument('--num_freqs', default=10, type=int) # Number of frequency for sin/cos

        # testing arguments
        parser.add_argument('--inference_loc', default="inference_out", type=str) # os.path.join(save_loc, inference_loc), where to cache inference results
        parser.add_argument('--gt_has_phase', default=0, type=bool_flag)  # image2reverb does not use gt phase for their GT when computing T60 error, and instead use random phase. If we use GT waveform (instead of randomizing the phase, we get lower T60 error)
        parser.add_argument('--baseline_mode', default="opus", type=str)  # Are we testing aac or opus? For baselines
        parser.add_argument('--interp_mode', default="nearest", choices=['linear', 'nearest'], type=str)  # interpolation mode. For baselines
        parser.add_argument('--fp16_interp', default=0, type=str)  # Use fp16 to save memory, essentially no change in results

        # visualization arguments
        parser.add_argument('--vis_ori', default=0, type=str)  # Choose an orientation to visualize, can be 0,1,2,3; corresponding to 0,90,180,270 degrees
        parser.add_argument('--room_grid_loc', default="room_grid_coors", type=str)  # where are the points for the room stored
        parser.add_argument('--room_feat_loc', default="room_feat_coors", type=str)  # where are the points for the room stored
        parser.add_argument('--room_grid_depth', default="room_depth_grid", type=str)  # room structure
        parser.add_argument('--room_scatter_depth', default="room_depth_scatter", type=str)  # room structure
        parser.add_argument('--vis_save_loc', default="loudness_img", type=str)
        parser.add_argument('--vis_feat_save_loc', default="feat_img", type=str)
        parser.add_argument('--depth_img_loc', default="depth_img", type=str)
        parser.add_argument('--net_feat_loc', default="network_feats_scatter", type=str) # a subset of the points in this folder is used for TSNE & linear fit
        parser.add_argument('--net_feat_loc2', default="network_feats_grid", type=str)
        parser.add_argument('--emitter_loc', default=[0.5, -3.0], type=list_float_flag)  # Where do we position the emitter? [0.5, -3.0] for apartment_1, [-0.2, 0.0] for apartment_2

    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.max_len = {"apartment_1": 101, "apartment_2": 86, "frl_apartment_2": 107, "frl_apartment_4": 103,
           "office_4": 78, "room_2": 84}

        # max_len_1024 = {"apartment_1": 52, "apartment_2": 44, "frl_apartment_2": 55, "frl_apartment_4": 53,
        #                 "office_4": 40, "room_2": 43}
        torch.manual_seed(0)
        # random.seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print()
        return self.opt