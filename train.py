import torch
torch.backends.cudnn.benchmark = True

from data_loading.sound_loader import soundsamples
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
from model.networks import kernel_residual_fc_embeds
from model.modules import embedding_module_log
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from time import time
from options import Options
import functools

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def worker_init_fn(worker_id, myrank_info):
    # print(worker_id + myrank_info*100, "SEED")
    np.random.seed(worker_id + myrank_info*100)

def train_net(rank, world_size, freeport, other_args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = freeport
    output_device = rank
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    pi = math.pi
    PIXEL_COUNT=other_args.pixel_count

    dataset = soundsamples(other_args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    ranked_worker_init = functools.partial(worker_init_fn, myrank_info=rank)
    sound_loader = torch.utils.data.DataLoader(dataset, batch_size=other_args.batch_size//world_size, shuffle=False, num_workers=3, worker_init_fn=ranked_worker_init, persistent_workers=True, sampler=train_sampler,drop_last=False)

    xyz_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)
    freq_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)

    auditory_net = kernel_residual_fc_embeds(input_ch=126, intermediate_ch=other_args.features, grid_ch=other_args.grid_features, num_block=other_args.layers, grid_gap=other_args.grid_gap, grid_bandwidth=other_args.bandwith_init, bandwidth_min=other_args.min_bandwidth, bandwidth_max=other_args.max_bandwidth, float_amt=other_args.position_float, min_xy=dataset.min_pos, max_xy=dataset.max_pos).to(output_device)

    if rank == 0:
        print("Dataloader requires {} batches".format(len(sound_loader)))

    start_epoch = 1
    load_opt = 0
    loaded_weights = False
    if other_args.resume:
        if not os.path.isdir(other_args.exp_dir):
            print("Missing save dir, exiting")
            dist.barrier()
            dist.destroy_process_group()
            return 1
        else:
            current_files = sorted(os.listdir(other_args.exp_dir))
            if len(current_files)>0:
                latest = current_files[-1]
                start_epoch = int(latest.split(".")[0]) + 1
                if rank == 0:
                    print("Identified checkpoint {}".format(latest))
                if start_epoch >= (other_args.epochs+1):
                    dist.barrier()
                    dist.destroy_process_group()
                    return 1
                map_location = 'cuda:%d' % rank
                weight_loc = os.path.join(other_args.exp_dir, latest)
                weights = torch.load(weight_loc, map_location=map_location)
                if rank == 0:
                    print("Checkpoint loaded {}".format(weight_loc))
                auditory_net.load_state_dict(weights["network"])
                loaded_weights = True
                if "opt" in weights:
                    load_opt = 1
                dist.barrier()
        if loaded_weights is False:
            print("Resume indicated, but no weights found!")
            dist.barrier()
            dist.destroy_process_group()
            exit()

    # We have conditional forward, must set find_unused_parameters to true
    ddp_auditory_net = DDP(auditory_net, find_unused_parameters=True, device_ids=[rank])
    criterion = torch.nn.MSELoss()
    orig_container = []
    grid_container = []
    for par_name, par_val in ddp_auditory_net.named_parameters():
        if "grid" in par_name:
            grid_container.append(par_val)
        else:
            orig_container.append(par_val)

    optimizer = torch.optim.AdamW([
        {'params': grid_container, 'lr': other_args.lr_init, 'weight_decay': 1e-2},
        {'params': orig_container, 'lr': other_args.lr_init, 'weight_decay': 0.0}], lr=other_args.lr_init, weight_decay=0.0)

    if load_opt:
        print("loading optimizer")
        optimizer.load_state_dict(weights["opt"])
        dist.barrier()


    if rank == 0:
        old_time = time()
    for epoch in range(start_epoch, other_args.epochs+1):
        total_losses = 0
        cur_iter = 0
        for data_stuff in sound_loader:
            gt = data_stuff[0].to(output_device, non_blocking=True)
            degree = data_stuff[1].to(output_device, non_blocking=True)
            position = data_stuff[2].to(output_device, non_blocking=True)
            non_norm_position = data_stuff[3].to(output_device, non_blocking=True)
            freqs = data_stuff[4].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
            times = data_stuff[5].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi

            with torch.no_grad():
                position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT, -1)
                freq_embed = freq_embedder(freqs)
                time_embed = time_embedder(times)

            total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)
            optimizer.zero_grad(set_to_none=False)
            try:
                output = ddp_auditory_net(total_in, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)
            except Exception as foward_exception:
                print(gt.shape, degree.shape, position.shape, freqs.shape, times.shape, position_embed.shape,
                      freq_embed.shape, time_embed.shape)
                print("Failure", foward_exception)
                continue
            loss = criterion(output, gt)
            if rank==0:
                total_losses += loss.detach()
                cur_iter += 1
            loss.backward()
            optimizer.step()
        decay_rate = other_args.lr_decay
        new_lrate_grid = other_args.lr_init * (decay_rate ** (epoch / other_args.epochs))
        new_lrate = other_args.lr_init * (decay_rate ** (epoch / other_args.epochs))

        par_idx = 0
        for param_group in optimizer.param_groups:
            if par_idx == 0:
                param_group['lr'] = new_lrate_grid
            else:
                param_group['lr'] = new_lrate
            par_idx += 1
        if rank == 0:
            avg_loss = total_losses.item() / cur_iter
            print("{}: Ending epoch {}, loss {}, time {}".format(other_args.exp_name, epoch, avg_loss, time() - old_time))
            old_time = time()
        if rank == 0 and (epoch%20==0 or epoch==1 or epoch>(other_args.epochs-3)):

            save_name = str(epoch).zfill(5)+".chkpt"
            save_dict = {}
            save_dict["network"] = ddp_auditory_net.module.state_dict()
            torch.save(save_dict, os.path.join(other_args.exp_dir, save_name))
    print("Wrapping up training {}".format(other_args.exp_name))
    dist.barrier()
    dist.destroy_process_group()
    return 1


if __name__ == '__main__':
    cur_args = Options().parse()
    exp_name = cur_args.exp_name
    exp_name_filled = exp_name.format(cur_args.apt)
    cur_args.exp_name = exp_name_filled
    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, creating...".format(cur_args.save_loc))
        os.mkdir(cur_args.save_loc)
    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir
    print("Experiment directory is {}".format(exp_dir))
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    world_size = cur_args.gpus
    myport = str(find_free_port())
    mp.spawn(train_net, args=(world_size, myport, cur_args), nprocs=world_size, join=True)
