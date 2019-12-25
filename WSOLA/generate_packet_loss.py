from random import random

import os
import sys
import argparse
import librosa
import soundfile as sf
import numpy as np

def packet_loss(x, loss_p, burst_p=0.5, step=160):
    state = 1 # 1: success, -1: fail
    total_frame = len(x)
    new_frame=[]
    loss_info = []
    buffer = np.zeros(step)

    for idx in range(0, total_frame, step):
        start_idx = idx
        end_idx = min(idx+step, total_frame)
        # state transition
        val = random()
        if state == 1:
            if val < loss_p:
                state = -1 # packet loss
            else:
                state = 1 # continue
        else:
            if val < burst_p:
                state = -1 # packet loss continue
            else:
                state = 1 # back to normal

        if state == 1:
            new_frame.extend(x[start_idx:end_idx])
        else:
            new_frame.extend(buffer[:end_idx-start_idx])
    return np.array(new_frame)

parser = argparse.ArgumentParser()
parser.add_argument("--loss_rate", default=0.3, type=float)
parser.add_argument("--output_dir", default=".\\packet_loss", type=str)
args = parser.parse_args()
loss_rate = args.loss_rate
output_dir = args.output_dir

if loss_rate <= 0 or loss_rate > 1:
    print("Wrong loss rate")
    sys.exit(1)

timit_path = ".\\ntimit"
dir_idx = len(timit_path)

for root, dirs, files in os.walk(timit_path):
    for file in files:
        path_input = os.path.join(root, file)
        path_output_dir = root.replace(timit_path, output_dir)
        os.makedirs(path_output_dir, exist_ok = True)
        path_output = os.path.join(path_output_dir, file)

        # os.path.split(root) = [_____\___\___\, last directory]
        speaker_id = os.path.split(root)[1]
        word_id = file.split('.')[0]

        x, sr = librosa.load(path_input, sr = 16000)
        x_with_packet_loss = packet_loss(x, loss_p = loss_rate)
        real_loss_rate = (1-(len(x_with_packet_loss)/len(x)))*100
        sf.write(path_output, x_with_packet_loss, sr, subtype='PCM_16')
        print(path_output)
        print(speaker_id + "_" + word_id, "Loss rate :", real_loss_rate)


# os.system("find " + timit_path + " -iname '*.wav' > wavlist.txt")
#
# with open("wavlist.txt", 'r') as f:
#     for line in f:
#         in_path = line[:-1]
#         out_path = output_dir + line[dir_idx:-1]
#         real_dir = "/".join(out_path.split("/")[:-1])
#         spk_id = out_path.split("/")[-2]
#         wrd_id = out_path.split("/")[-1].split(".")[0]
#         os.system("mkdir -p " + real_dir)
#         cmd = "cat "+in_path+" > "+out_path
#         os.system(cmd)
#         x, sr = librosa.load(out_path, sr=16000)
#         new_x = packet_loss(x, loss_p=loss_rate)
#
#         # show drop rate
#         real_loss_rate = (1-(len(new_x)/len(x)))*100
#         print(spk_id+"_"+wrd_id, "Loss rate :", real_loss_rate)
#         # Write out audio as 16bit PCM WAV
#         sf.write(out_path, new_x, sr, subtype='PCM_16')
# sys.exit()
