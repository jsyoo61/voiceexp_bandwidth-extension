import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from wsola import wsola

########## hyperparameters ##########
input_dir = '.\\packet_loss'
output_dir = '.\\reconstructed'

for root, dirs, files in os.walk(input_dir):
    for file in files:
        path_input = os.path.join(root, file)
        path_output_dir = root.replace(input_dir, output_dir)
        os.makedirs(path_output_dir, exist_ok = True)
        path_output = os.path.join(path_output_dir, file)
        print(path_output)

        # os.path.split(root) = [_____\___\___\, last directory]
        speaker_id = os.path.split(root)[1]
        word_id = file.split('.')[0]

        x_with_packet_loss, sr = librosa.load(path_input, sr = 16000)
        # plt.plot(x_with_packet_loss)
        # plt.show()
        x_reconstructed = wsola(x_with_packet_loss, sr = sr)
        real_loss_rate = (1-(len(x_reconstructed)/len(x_with_packet_loss)))*100
        # plt.plot(x_reconstructed)
        # plt.show()
        sf.write(path_output, x_reconstructed, sr, subtype='PCM_16')
        print(speaker_id + "_" + word_id, "Loss rate :", real_loss_rate)
