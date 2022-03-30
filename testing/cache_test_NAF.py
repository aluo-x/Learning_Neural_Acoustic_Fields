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

def to_torch(input_arr):
    return input_arr[None]

def test_net(rank, other_args):
    pi = math.pi
    output_device = rank
    print("creating dataset")
    dataset = soundsamples(other_args)
    xyz_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)
    freq_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)
    auditory_net = kernel_residual_fc_embeds(input_ch=126, intermediate_ch=other_args.features, grid_ch=other_args.grid_features, num_block=other_args.layers, grid_gap=other_args.grid_gap, grid_bandwidth=other_args.bandwith_init, bandwidth_min=other_args.min_bandwidth, bandwidth_max=other_args.max_bandwidth, float_amt=other_args.position_float, min_xy=dataset.min_pos, max_xy=dataset.max_pos).to(output_device)

    loaded_weights = False
    current_files = sorted(os.listdir(other_args.exp_dir))
    if len(current_files)>0:
        latest = current_files[-1]
        print("Identified checkpoint {}".format(latest))
        map_location = 'cuda:%d' % rank
        weight_loc = os.path.join(other_args.exp_dir, latest)
        weights = torch.load(weight_loc, map_location=map_location)
        print("Checkpoint loaded {}".format(weight_loc))
        auditory_net.load_state_dict(weights["network"])
        loaded_weights = True
    if loaded_weights is False:
        print("Weights not found")

    auditory_net.eval()
    container = dict()
    save_name = os.path.join(other_args.result_output_dir, other_args.apt+"_NAF.pkl")
    container["mean_std"] = (dataset.std, dataset.mean)
    with torch.no_grad():
        for ori in [0, 1, 2, 3]:
            num_sample_test = len(dataset.sound_files_test[["0", "90", "180", "270"][ori]])
            ori_offset = 0
            print("Total {} for orientation {}".format(num_sample_test, str(ori)))
            for test_id in range(num_sample_test):
                ori_offset += 1
                if ori_offset%100 == 0:
                    print("Currently on {}".format(ori_offset))
                data_stuff = dataset.get_item_test(ori, test_id)
                gt = to_torch(data_stuff[0])
                degree = torch.Tensor([data_stuff[1]]).to(output_device, non_blocking=True).long()
                position = to_torch(data_stuff[2]).to(output_device, non_blocking=True)
                non_norm_position = to_torch(data_stuff[3]).to(output_device, non_blocking=True)
                freqs = to_torch(data_stuff[4]).to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
                times = to_torch(data_stuff[5]).to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
                PIXEL_COUNT = gt.shape[-1]
                position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT, -1)
                freq_embed = freq_embedder(freqs)
                time_embed = time_embedder(times)
                total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)
                output = auditory_net(total_in, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)
                myout = output.cpu().numpy()
                myout = myout.reshape(1, 2, dataset.sound_size[1], dataset.sound_size[2])
                mygt = gt.numpy()
                mygt = mygt.reshape(1, 2, dataset.sound_size[1], dataset.sound_size[2])
                container["{}_{}".format(ori, dataset.sound_name)]= [myout, mygt, dataset.sound_size]
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
    world_size = cur_args.gpus
    test_ = test_net(0, cur_args)
    ## Uncomment to run all rooms
    # for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
    #     cur_args.apt = apt
    #     exp_name = cur_args.exp_name
    #     exp_name_filled = exp_name.format(cur_args.apt)
    #     cur_args.exp_name = exp_name_filled
    #
    #     exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    #     cur_args.exp_dir = exp_dir
    #
    #     result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    #     cur_args.result_output_dir = result_output_dir
    #     if not os.path.isdir(result_output_dir):
    #         os.mkdir(result_output_dir)
    #
    #     if not os.path.isdir(cur_args.save_loc):
    #         print("Save directory {} does not exist, need checkpoint folder...".format(cur_args.save_loc))
    #         exit()
    #     if not os.path.isdir(cur_args.exp_dir):
    #         print("Experiment {} does not exist, need experiment folder...".format(cur_args.exp_name))
    #         exit()
    #     print("Experiment directory is {}".format(exp_dir))
    #     world_size = cur_args.gpus
    #     test_ = test_net(0, cur_args)
