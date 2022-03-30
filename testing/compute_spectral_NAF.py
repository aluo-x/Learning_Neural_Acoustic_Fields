import pickle
from inspect import getsourcefile
import os

import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from testing.test_utils import spectral
from options import Options

cur_args = Options().parse()
# for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
#     cur_args.apt = apt
exp_name = cur_args.exp_name
exp_name_filled = exp_name.format(cur_args.apt)
cur_args.exp_name = exp_name_filled

exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
cur_args.exp_dir = exp_dir

result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
cur_args.result_output_dir = result_output_dir

data_path = os.path.join(cur_args.result_output_dir, cur_args.apt+"_NAF.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

all_keys = list(data.keys())
std, mean = data["mean_std"]
std = std.numpy()
mean = mean.numpy()

loss = 0
total = 0
for k in all_keys:
    if k == 'mean_std':
        continue
    cur_data = data[k]
    actual_spec_len = cur_data[0].shape[-1]
    std_ = std[:,:,:actual_spec_len]
    mean_ = mean[:,:,:actual_spec_len]
    net_out = cur_data[0][0]*std_ + mean_
    gt_out = cur_data[1][0]*std_ + mean_
    loss += spectral(net_out, gt_out)
    total += 1.0

mean_loss = loss/total
print("{} the spectral loss is {}".format(cur_args.apt, mean_loss))