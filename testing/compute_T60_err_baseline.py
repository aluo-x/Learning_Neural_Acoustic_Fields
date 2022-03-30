import pickle
from inspect import getsourcefile
import os

# import matplotlib.pyplot as plt
import numpy as np
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from testing.test_utils import get_wave, compute_t60, load_audio, get_spec
from options import Options
import h5py
# import subprocess

cur_args = Options().parse()
exp_name = cur_args.exp_name
spec_getter = get_spec()
# # Uncomment to run all rooms
# for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
#     for baseline in ["aac", "opus"]:
#         for mode in ["nearest", "linear"]:
#             cur_args.baseline_mode = baseline
#             cur_args.interp_mode = mode
#             cur_args.apt = apt
apt = cur_args.apt
baseline = cur_args.baseline_mode
cur_args.gt_has_phase = True  # Generally 0.5% difference regardless of which one you use, but True is faster (since no inverse STFT)
exp_name_filled = exp_name.format(cur_args.apt)
cur_args.exp_name = exp_name_filled

exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
cur_args.exp_dir = exp_dir

result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
cur_args.result_output_dir = result_output_dir

data_path = os.path.join(cur_args.result_output_dir,cur_args.apt + "_{}_{}.pkl".format(cur_args.baseline_mode, cur_args.interp_mode))
with open(data_path, "rb") as f:
    data = pickle.load(f)
full_path = os.path.join(cur_args.spec_base, cur_args.apt + ".h5")
sound_data_gt = h5py.File(full_path, 'r')
all_keys = sorted(list(data.keys()))
loss = 0
total = 0
all_t60 = []
for k in all_keys:
    baseline_wav = data[k].T
    if not cur_args.gt_has_phase:
        gt_out = sound_data_gt[k][:].astype(np.single)
        gt_wav = get_wave(np.clip(np.exp(gt_out)-1e-3, 0.0, 10000.00))
    else:
        k_split = k.split("_")
        orientation = k_split[0]
        audio_file_name = os.path.join(cur_args.wav_base, apt, orientation, "{}.wav".format("_".join(k_split[1:])))
        gt_wav = load_audio(audio_file_name)
    t60s = compute_t60(gt_wav.astype(np.double), baseline_wav.astype(np.double))
    all_t60.append(t60s)
t60s_np = np.array(all_t60)
diff = np.abs(t60s_np[:,2:]-t60s_np[:,:2])/np.abs(t60s_np[:,:2])
mask = np.any(t60s_np<-0.5, axis=1)
diff = np.mean(diff, axis=1)
diff[mask] = 1
print("{} total invalids out of {}".format(np.sum(mask), mask.shape[0]), apt, cur_args.interp_mode, baseline, np.mean(diff)*100)