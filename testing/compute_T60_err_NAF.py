import pickle
from inspect import getsourcefile
import os
import numpy as np
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from testing.test_utils import get_wave, compute_t60, load_audio
from options import Options

cur_args = Options().parse()
exp_name = cur_args.exp_name
# for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
#     cur_args.apt = apt
apt = cur_args.apt
cur_args.gt_has_phase = False # Match image2reverb behavior, they also use False
print(apt)
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
all_t60 = []
for k in all_keys:
    if k == 'mean_std':
        continue
    cur_data = data[k]
    actual_spec_len = cur_data[0].shape[-1]
    std_ = std[:,:,:actual_spec_len]
    mean_ = mean[:,:,:actual_spec_len]
    net_out = cur_data[0][0] * std_ + mean_
    net_wav = get_wave(np.clip(np.exp(net_out)-1e-3, 0.0, 10000.00))
    if not cur_args.gt_has_phase:
        gt_out = cur_data[1][0] * std_ + mean_
        gt_wav = get_wave(np.clip(np.exp(gt_out)-1e-3, 0.0, 10000.00))
    else:
        orientation = str([0, 90, 180, 270][int(k.split("_")[0])])
        node_names = k.split("_")[1].replace("[", "").replace("]", "").replace("'", "").split(",")
        first = str(int(node_names[0]))
        second = str(int(node_names[1]))
        audio_file_name = os.path.join(cur_args.wav_base, apt, orientation, "{}_{}.wav".format(first, second))
        gt_wav = load_audio(audio_file_name)
    t60s = compute_t60(gt_wav, net_wav)
    all_t60.append(t60s)

t60s_np = np.array(all_t60)
diff = np.abs(t60s_np[:,2:]-t60s_np[:,:2])/np.abs(t60s_np[:,:2])
mask = np.any(t60s_np<-0.5, axis=1)
diff = np.mean(diff, axis=1)
diff[mask] = 1
print("{} total invalids out of {}".format(np.sum(mask), mask.shape[0]))
print(np.mean(diff)*100)