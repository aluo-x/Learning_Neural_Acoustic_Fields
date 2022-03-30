from inspect import getsourcefile
import numpy as np
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])


import pickle
import os
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
import math
from options import Options
from time import time
from scipy.io.wavfile import read

def test_baseline(other_args):
    print("Getting the keys of the test set")
    train_test_split_path = os.path.join(other_args.split_loc, other_args.apt+"_complete.pkl")
    with open(train_test_split_path, "rb") as train_test_file_obj:
        train_test_split = pickle.load(train_test_file_obj)

    train_split = train_test_split[0]
    test_split = train_test_split[1]

    interp_mode = other_args.interp_mode
    coor_base = other_args.coor_base
    coor_path = os.path.join(coor_base, other_args.apt, "points.txt")
    coors = np.loadtxt(coor_path)[:,1:][:,[0,1]]
    coors[:, 1] = -coors[:, 1]
    coors = coors.astype(np.single)
    if other_args.baseline_mode == "opus":
        train_folder = os.path.join(other_args.opus_write_path, other_args.apt)
    elif other_args.baseline_mode == "aac":
        train_folder = os.path.join(other_args.aac_write_path, other_args.apt)
    else:
        print("Baseline mode must be of (aac, opus)")
        exit()

    # train_split = {"0":[], "90":[], "180":[], "270":[]}
    container = dict()
    save_name = os.path.join(other_args.result_output_dir, other_args.apt + "_{}_{}.pkl".format(other_args.baseline_mode, interp_mode))
    if os.path.isfile(save_name):
        return 1
    for ori in ["0", "90", "180", "270"]:
        train_coors = []
        train_data = []
        print("Processing orientation {}".format(ori))
        train_keys = train_split[ori]
        print("Loading sound data")
        ori_str = os.path.join(train_folder, ori)
        for train_str in train_keys:
            cur_item_trainer = train_str
            pos_id_train = cur_item_trainer.split("_")
            listener, emitter = pos_id_train
            listener_pos = coors[int(listener)]
            emitter_pos = coors[int(emitter)]
            total_pos = np.concatenate((listener_pos, emitter_pos), axis=0)
            train_wav_path = os.path.join(ori_str, train_str + ".wav")
            try:
                gt_data = read(train_wav_path)
            except:
                # Sometimes the encoder/decoder fails on a couple files. Just skip them.
                print("Missing ", train_wav_path)
                continue
            if not np.all(np.isfinite(gt_data[1])):
                continue
            train_coors.append(total_pos)
            if other_args.fp16_interp and 0:
                train_data.append(gt_data[1].astype(np.half))
            else:
                train_data.append(gt_data[1].astype(np.single))

        max_length = max([asd.shape[0] for asd in train_data])
        train_data = np.array([np.pad(train_data_x, ((0, max_length - train_data_x.shape[0]), (0, 0))) for train_data_x in train_data])
        train_data.setflags(write=False)
        train_coors = np.array(train_coors)
        train_coors.setflags(write=False)
        print("Finished loading data for orientation {}".format(ori))
        print("Starting {} interp object construction, go grab a coffee".format(interp_mode))
        old_t = time()
        if interp_mode == "linear":
            print("Constructing linear interpolation engine")
            interp_engine = LinearNDInterpolator(points=train_coors, values=train_data, fill_value=0.0,rescale=False)
        elif interp_mode == "nearest":
            print("Constructing nearest interpolation engine")
            interp_engine = NearestNDInterpolator(x=train_coors, y=train_data)
        else:
            print("Interpolation mode must be of (linear, nearest)")
            exit()
        print(train_data.shape, )
        print("Completed interp object contruction in {} seconds".format(str(time()-old_t)))
        for test_str in list(test_split[ori]):
            pos_id_train = test_str.split("_")
            listener, emitter = pos_id_train
            listener_pos = coors[int(listener)]
            emitter_pos = coors[int(emitter)]
            total_pos = np.concatenate((listener_pos, emitter_pos), axis=0)
            out = interp_engine(total_pos)[0]
            container["{}_{}".format(ori, test_str)] = out.astype(np.single) + 0.0
    with open(save_name, "wb") as saver_file_obj:
        pickle.dump(container, saver_file_obj)
        print("Results saved to {}".format(save_name))
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
    test_ = test_baseline(cur_args)

# #Uncomment to run every room
# for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
#     cur_args.apt = apt
#     for baseline_mode in ["opus", "aac"]:
#         cur_args.baseline_mode = baseline_mode
#         for i_mode in ["nearest", "linear"]:
#             cur_args.interp_mode = i_mode
#             test_ = test_baseline(cur_args)
