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
import random, string
from sklearn.metrics import explained_variance_score

def to_torch(input_arr):
    return input_arr[None]

def lin_probe_feat(rank, other_args):
    pi = math.pi
    output_device = rank

    with open(os.path.join(other_args.save_loc, other_args.net_feat_loc, other_args.apt+"_features.pkl"),"rb") as feat_file_obj:
        train_feats_superset = list(pickle.load(feat_file_obj))

    train_points = np.array([_[1] for _ in train_feats_superset])
    train_feats_superset = np.array([np.array(_[0]).flatten() for _ in train_feats_superset])

    with open(os.path.join(other_args.save_loc, other_args.net_feat_loc2, other_args.apt+"_features.pkl"),"rb") as test_file_obj:
        test_feats = list(pickle.load(test_file_obj))

    test_points = np.array([_[1] for _ in test_feats])
    test_feats = np.array([np.array(_[0]).flatten() for _ in test_feats])

    with open(os.path.join("./metadata", other_args.room_scatter_depth, other_args.apt + ".pkl"),"rb") as train_file_obj:
        train_depth = list(pickle.load(train_file_obj))
    with open(os.path.join("./metadata", other_args.room_grid_depth, other_args.apt + ".pkl"),"rb") as test_file_obj:
        print(os.path.join("./metadata", other_args.room_grid_depth, other_args.apt + ".pkl"))
        test_depth = np.array(list(pickle.load(test_file_obj)))

    print("Creating linear probe layer")
    linear_probe = torch.nn.Linear(512*5, 1).cuda(rank)

    total_len = int(len(train_feats_superset) * 2.0 / 5.0)

    training_data = torch.from_numpy(train_feats_superset).float()[:total_len]
    training_data = training_data.cuda(rank)

    opt = torch.optim.AdamW([*linear_probe.parameters()], lr=1e-4)

    labels_new = torch.from_numpy(np.array(train_depth)).float()[:, None][:total_len]
    labels_new = labels_new.cuda(rank)
    criterion = torch.nn.MSELoss()

    for k in range(50000):
        y_pred = linear_probe(training_data)
        opt.zero_grad()
        loss = criterion(y_pred, labels_new)
        loss.backward()
        opt.step()
    testing_data = torch.from_numpy(test_feats).cuda(rank)
    linear_probe.eval()
    with torch.no_grad():
        predicted_depth = linear_probe(testing_data).cpu().detach().numpy()
    plt.scatter(test_points[:,0], test_points[:, 1], c=predicted_depth[:,0], vmin=np.min(test_depth), vmax=np.max(test_depth))
    plt.axis('equal')
    plt.axis('off')
    # plt.show()
    plt.savefig(os.path.join(other_args.save_loc, other_args.depth_img_loc, other_args.apt + "_NAF.png"))
    plt.close()

    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_depth, vmin=np.min(test_depth),
                vmax=np.max(test_depth))
    plt.axis('equal')
    plt.axis('off')
    # plt.show()
    plt.savefig(os.path.join(other_args.save_loc, other_args.depth_img_loc, other_args.apt + "_GT.png"))
    print("Explained variance: ", explained_variance_score(test_depth, predicted_depth[:,0]))
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
    test_ = lin_probe_feat(0, cur_args)
