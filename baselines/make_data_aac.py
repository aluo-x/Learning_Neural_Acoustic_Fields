import os
import subprocess
import shutil
from joblib import Parallel, delayed
import pickle

from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from options import Options

def process(name_tup):
    outsize=0
    worked = 0
    e = None
    try:
        encode_cmd = "{} -i {} -c:a aac -b:a 24k {}".format(name_tup[-1], name_tup[0], name_tup[1])
        decode_cmd = "{} -i {} -c:a pcm_f32le -ar 22050 {}".format(name_tup[-1], name_tup[1], name_tup[2])

        subprocess.call(encode_cmd, timeout=20, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, shell=True)
        subprocess.call(decode_cmd, timeout=20, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, shell=True)
        shutil.copyfile(name_tup[2], name_tup[3])
        outsize = os.path.getsize(name_tup[1])
        os.remove(name_tup[1])
        os.remove(name_tup[2])
        worked = 1
    except Exception as e:
        attempts = 1

        while worked == 0 and attempts<3:
            try:
                encode_cmd = "{} -i {} -c:a aac -b:a 24k {}".format(name_tup[-1], name_tup[0], name_tup[1])
                decode_cmd = "{} -i {} -c:a pcm_f32le -ar 22050 {}".format(name_tup[-1], name_tup[1], name_tup[2])

                subprocess.call(encode_cmd, timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
                subprocess.call(decode_cmd, timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
                shutil.copyfile(name_tup[2], name_tup[3])
                outsize = os.path.getsize(name_tup[1])
                os.remove(name_tup[1])
                os.remove(name_tup[2])
                worked = 1
            except:
                pass
            attempts += 1
    if worked == 0:
        try:
            print(e)
        except:
            print("some failure")
    try:
        os.remove(name_tup[1])
        os.remove(name_tup[2])
    except:
        pass
    return outsize

if __name__ == '__main__':
    cur_args = Options().parse()
    raw_path = cur_args.wav_base
    write_path = cur_args.aac_write_path
    room_names = os.listdir(raw_path)

    for room_name in room_names:
        print("processing {}".format(room_name))
        room_path = os.path.join(raw_path, room_name)
        aac_room_path = os.path.join(write_path, room_name)
        if not os.path.exists(aac_room_path):
            os.mkdir(aac_room_path)
        zz = 0
        total_container = []
        total_temp = []
        total_out = []
        final_loc = []
        root = cur_args.ramdisk_path
        size_new = 0
        k_name_offset = room_name
        for orientation in ["0", "90", "180", "270"]:
            ori_path = os.path.join(room_path, orientation)
            aac_ori_path = os.path.join(aac_room_path, orientation)
            if not os.path.exists(aac_ori_path):
                os.mkdir(aac_ori_path)
            files = sorted(os.listdir(ori_path))
            files = [_ for _ in files if "wav" in _]
            if orientation == "0":
                print("Found {} files".format(str(len(files) * 4)))
            for ff in files:
                zz += 1
                cur_file = os.path.join(ori_path, ff)
                if os.path.getsize(cur_file) < 1024:
                    print(cur_file)
                    print("skipping")
                    continue
                if os.path.exists(os.path.join(aac_ori_path, ff)):
                    continue
                total_container.append(cur_file)
                total_temp.append(os.path.join(root, str(orientation) + k_name_offset + str(zz) + ".m4a"))
                total_out.append(os.path.join(root, str(orientation) + k_name_offset + str(zz) + ".wav"))
                final_loc.append(os.path.join(aac_ori_path, ff))
        all_total = list(zip(total_container, total_temp, total_out, final_loc, [cur_args.ffmpeg]*len(final_loc)))
        out = Parallel(n_jobs=16, verbose=5)(delayed(process)(i) for i in all_total)
        with open(os.path.join(aac_room_path, "sizes_aac.pkl"), "wb") as writer:
            pickle.dump(out, writer)
