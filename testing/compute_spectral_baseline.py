import pickle
from inspect import getsourcefile
import os
import h5py
import numpy as np

import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from testing.test_utils import spectral, get_spec, load_audio
from options import Options

cur_args = Options().parse()
exp_name = cur_args.exp_name
# for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
#     for baseline in ["aac", "opus"]:
#         for mode in ["nearest", "linear"]:
#             cur_args.baseline_mode = baseline
#             cur_args.interp_mode = mode
#             cur_args.apt = apt
exp_name_filled = exp_name.format(cur_args.apt)
cur_args.exp_name = exp_name_filled

exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
cur_args.exp_dir = exp_dir

result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
cur_args.result_output_dir = result_output_dir

data_path = os.path.join(cur_args.result_output_dir, cur_args.apt+"_{}_{}.pkl".format(cur_args.baseline_mode, cur_args.interp_mode))
with open(data_path, "rb") as f:
    data = pickle.load(f)

full_path = os.path.join(cur_args.spec_base, cur_args.apt+".h5")
sound_data_gt = h5py.File(full_path, 'r')

loss = 0
total = 0
all_keys = list(data.keys())
spec_getter = get_spec()

for k in all_keys:
    spec_baseline = spec_getter.transform(data[k].T)
    spec_gt = sound_data_gt[k][:]
    # we padded the sound file prior to interpolation
    # crop here to correct length & match NAF spectral behavior
    spec_len = min(spec_gt.shape[-1], spec_baseline.shape[-1])
    loss += spectral(spec_baseline[...,:spec_len], spec_gt[..., :spec_len])
    total += 1.0

mean_loss = loss/total
print("{} the spectral loss is {}".format(cur_args.apt, mean_loss), cur_args.apt, cur_args.interp_mode, cur_args.baseline_mode)