import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from scipy.io.wavfile import write as wavwrite
# from scipy.io.wavfile import read as wavread

from util import sampling
from network import CleanUNet

import torchaudio

# with open("configs/DNS-large-high.json") as f:
#     data = f.read()
# config = json.loads(data)
network_config = {
        "channels_input": 1,
        "channels_output": 1,
        "channels_H": 64,
        "max_H": 768,
        "encoder_n_layers": 8,
        "kernel_size": 4,
        "stride": 2,
        "tsfm_n_layers": 5, 
        "tsfm_n_head": 8,
        "tsfm_d_model": 512, 
        "tsfm_d_inner": 2048
    }
def denoise(filename, output_directory='results', sample_rate=16000, model_path = 'exp/DNS-large-high/checkpoint/pretrained.pkl'):
    net = CleanUNet(**network_config).cpu()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    with torch.no_grad():

        noisy_audio, sample_rate = torchaudio.load(filename)
        assert sample_rate==16000, "sample rate have to be 16000"
        noisy_audio=noisy_audio.cpu()
        final = np.zeros(noisy_audio.shape[1])

        batch_size = 30 * sample_rate

        for i in tqdm(range(0, noisy_audio.shape[1],batch_size)):
            next_size = batch_size + i
            if batch_size +i > noisy_audio.shape[1]:
                next_size = noisy_audio.shape[1]
            batch = noisy_audio[0:2,i: next_size]
            generated_audio = sampling(net, batch)
            final[i: next_size]=generated_audio[0]

        wavwrite(os.path.join(output_directory, 'enhanced_{}'.format(filename.split('/')[-1])), 
                    sample_rate,
                    final)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='denoised file', required=True)
    parser.add_argument('-d', '--directory', type=str, default='results',
                        help='output directory')
    parser.add_argument('-m', '--model_path', type=str, default='exp/DNS-large-high/checkpoint/pretrained.pkl',
                        help='model path')
    args = parser.parse_args()

    denoise(args.filename, output_directory=args.directory,model_path=args.model_path)