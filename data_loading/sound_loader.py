import numpy.random
import torch
import os
import pickle
import numpy as np
import random
import h5py
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def listdir(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

def join(*paths):
    return os.path.join(*paths)


class soundsamples(torch.utils.data.Dataset):
    def __init__(self, arg_stuff):
        coor_base = arg_stuff.coor_base
        spec_base = arg_stuff.spec_base
        mean_std_base = arg_stuff.mean_std_base
        minmax_base = arg_stuff.minmax_base
        room_name = arg_stuff.apt
        num_samples = arg_stuff.pixel_count

        coor_path = os.path.join(coor_base, room_name, "points.txt")
        max_len = arg_stuff.max_len
        self.max_len = max_len[room_name]
        full_path = os.path.join(spec_base, room_name+".h5")

        print("Caching the room coordinate indices, this will take a while....")
        # See https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643
        self.sound_data = []
        self.sound_data = h5py.File(full_path, 'r')
        self.sound_keys = list(self.sound_data.keys())
        self.sound_data.close()
        print("Completed room coordinate index caching")
        self.sound_data = None
        self.full_path = full_path

        files = [(mykey.split("_")[0], "_".join(mykey.split("_")[1:])) for mykey in self.sound_keys]
        self.sound_files = {"0": [], "90": [], "180": [], "270": []}
        self.sound_files_test = {"0": [], "90": [], "180": [], "270": []}

        train_test_split_path = os.path.join(arg_stuff.split_loc, arg_stuff.apt + "_complete.pkl")
        with open(train_test_split_path, "rb") as train_test_file_obj:
            train_test_split = pickle.load(train_test_file_obj)
        # use train test split

        self.sound_files = train_test_split[0]
        self.sound_files_test = train_test_split[1]

        with open(os.path.join(mean_std_base, room_name+".pkl"), "rb") as mean_std_ff:
            mean_std = pickle.load(mean_std_ff)
            print("Loaded mean std")
        self.mean = torch.from_numpy(mean_std[0]).float()[None]
        self.std = 3.0 * torch.from_numpy(mean_std[1]).float()[None]

        with open(coor_path, "r") as f:
            lines = f.readlines()
        coords = [x.replace("\n", "").split("\t") for x in lines]
        self.positions = dict()
        for row in coords:
            readout = [float(xyz) for xyz in row[1:]]
            self.positions[row[0]] = [readout[0], -readout[1]]

        with open(os.path.join(minmax_base, room_name+"_minmax.pkl"), "rb") as min_max_loader:
            min_maxes = pickle.load(min_max_loader)
            self.min_pos = min_maxes[0][[0, 2]]
            self.max_pos = min_maxes[1][[0, 2]]
            # This is since dimension 0 and 2 are floor plane

        # values = np.array(list(self.positions.values()))
        self.num_samples = num_samples
        self.pos_reg_amt = arg_stuff.reg_eps

    def __len__(self):
        # return number of samples for a SINGLE orientation
        return len(list(self.sound_files.values())[0])

    def __getitem__(self, idx):
        loaded = False
        orientations = ["0", "90", "180", "270"]
        while not loaded:
            try:
                orientation_idx = random.randint(0, 3)
                orientation = orientations[orientation_idx]

                if self.sound_data is None:
                    self.sound_data = h5py.File(self.full_path, 'r')

                pos_id = self.sound_files[orientation][idx]
                query_str = orientation + "_" + pos_id

                spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()
                position = (pos_id.split(".")[0]).split("_")
                spec_data = spec_data[:,:,:self.max_len]

                if random.random()<0.1:
                    # np.log(1e-3) = -6.90775527898213
                    spec_data = torch.nn.functional.pad(spec_data, pad=[0, self.max_len-spec_data.shape[2], 0, 0, 0, 0], value=-6.90775527898213)

                actual_spec_len = spec_data.shape[2]
                spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
                # 2, freq, time
                sound_size = spec_data.shape
                selected_time = np.random.randint(0, sound_size[2], self.num_samples)
                selected_freq = np.random.randint(0, sound_size[1], self.num_samples)
                degree = orientation_idx

                non_norm_start = (np.array(self.positions[position[0]])[:2] + np.random.normal(0, 1, 2)*self.pos_reg_amt)
                non_norm_end = (np.array(self.positions[position[1]])[:2]+ np.random.normal(0, 1, 2)*self.pos_reg_amt)
                start_position = (torch.from_numpy((non_norm_start - self.min_pos)/(self.max_pos-self.min_pos))[None] - 0.5) * 2.0
                start_position = torch.clamp(start_position, min=-1.0, max=1.0)

                end_position = (torch.from_numpy((non_norm_end - self.min_pos)/(self.max_pos-self.min_pos))[None] - 0.5) * 2.0
                end_position = torch.clamp(end_position, min=-1.0, max=1.0)

                total_position = torch.cat((start_position, end_position), dim=1).float()

                total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

                selected_total = spec_data[:,selected_freq,selected_time]
                loaded = True

            except Exception as e:
                print(query_str)
                print(e)
                print("Failed to load sound sample")

        return selected_total, degree, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0

    def get_item_teaser(self, orientation_idx, reciever_pos, source_pos):
        selected_time = np.arange(0, self.max_len)
        selected_freq = np.arange(0, 256)
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)

        degree = orientation_idx

        non_norm_start = np.array(reciever_pos)
        non_norm_end = np.array(source_pos)
        total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

        start_position = (torch.from_numpy((non_norm_start - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        start_position = torch.clamp(start_position, min=-1.0, max=1.0)
        end_position = (torch.from_numpy((non_norm_end - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        end_position = torch.clamp(end_position, min=-1.0, max=1.0)
        total_position = torch.cat((start_position, end_position), dim=1).float()

        return degree, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0

    def get_item_test(self, orientation_idx, idx):
        orientations = ["0", "90", "180", "270"]
        orientation = orientations[orientation_idx]
        selected_files = self.sound_files_test
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')
        pos_id = selected_files[orientation][idx]
        query_str = orientation + "_" + pos_id
        spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()

        position = (pos_id.split(".")[0]).split("_")

        spec_data = spec_data[:, :, :self.max_len]
        actual_spec_len = spec_data.shape[2]

        spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
        # 2, freq, time
        sound_size = spec_data.shape
        self.sound_size = sound_size
        self.sound_name = position
        selected_time = np.arange(0, sound_size[2])
        selected_freq = np.arange(0, sound_size[1])
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)

        degree = orientation_idx

        non_norm_start = np.array(self.positions[position[0]])[:2]
        non_norm_end = np.array(self.positions[position[1]])[:2]
        start_position = (torch.from_numpy((non_norm_start - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        start_position = torch.clamp(start_position, min=-1.0, max=1.0)
        end_position = (torch.from_numpy((non_norm_end - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        end_position = torch.clamp(end_position, min=-1.0, max=1.0)
        total_position = torch.cat((start_position, end_position), dim=1).float()
        total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

        selected_total = spec_data[:, selected_freq, selected_time]
        return selected_total, degree, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0


