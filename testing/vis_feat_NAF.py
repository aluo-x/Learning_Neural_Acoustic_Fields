import matplotlib.pyplot as plt
import torch
torch.backends.cudnn.benchmark = True
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from data_loading.sound_loader import soundsamples
import pickle
import os
from model.networks import kernel_residual_fc_embeds
from model.modules import embedding_module_log
import math
from options import Options
import numpy as np
from librosa.feature import rms
import random, string
from sklearn.manifold import TSNE

def to_torch(input_arr):
    return input_arr[None]

def vis_feat(rank, other_args):
    pi = math.pi
    output_device = rank
    with open(os.path.join(other_args.save_loc, other_args.net_feat_loc, other_args.apt+"_features.pkl"),"rb") as feat_file_obj:
        plot_feats = list(pickle.load(feat_file_obj))

    features = np.array([np.array(_[0]).flatten() for _ in plot_feats])
    points = np.array([_[1] for _ in plot_feats])
    feat_reducer = TSNE(n_components=3)
    feat_vis = feat_reducer.fit_transform(features)

    # More pretty as suggested
    feat_vis = np.abs(feat_vis)
    feat_vis = feat_vis-np.min(feat_vis)
    feat_vis = feat_vis/np.max(feat_vis)

    total_len = int(len(points)*2.0/5.0) # Use fewer points for visualization, or else too dense
    plt.rcParams['figure.dpi'] = 150
    plt.scatter(points[:total_len, 0], points[:total_len, 1], c=feat_vis[:total_len], s=30, alpha=0.9, edgecolors=None, linewidths=0.0)
    plt.axis('equal')
    plt.axis('off')
    # plt.show()
    plt.savefig(os.path.join(other_args.save_loc, other_args.vis_feat_save_loc, other_args.apt+"_features.png"))
    return 1

if __name__ == '__main__':
    cur_args = Options().parse()
    exp_name = cur_args.exp_name
    exp_name_filled = exp_name.format(cur_args.apt)
    cur_args.exp_name = exp_name_filled

    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir

    result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    cur_args.result_output_dir = result_output_dir
    if not os.path.isdir(result_output_dir):
        os.mkdir(result_output_dir)

    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, need checkpoint folder...".format(cur_args.save_loc))
        exit()
    if not os.path.isdir(cur_args.exp_dir):
        print("Experiment {} does not exist, need experiment folder...".format(cur_args.exp_name))
        exit()
    print("Experiment directory is {}".format(exp_dir))
    world_size = cur_args.gpus
    test_ = vis_feat(0, cur_args)
