import os
import argparse
import json
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread

from dataset import load_CleanNoisyPairDataset
from util import rescale, find_max_epoch, print_size, sampling
from network import CleanUNet

import torchaudio

with open("configs/DNS-large-high.json") as f:
    data = f.read()
config = json.loads(data)

output_directory='results'
ckpt_iter='pretrained'
subset='testing'
dump=True
sample_rate=16000
exp_path="exp"

net = CleanUNet(**config['network_config']).cpu()


model_path = 'exp/DNS-large-high/checkpoint/pretrained.pkl'

checkpoint = torch.load(model_path, map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

print("loaded model")

filename="dns/audio.wav"

all_generated_audio = []

noisy_audio, sample_rate = torchaudio.load(filename)
noisy_audio=noisy_audio.cpu()

generated_audio = sampling(net, noisy_audio)


wavwrite(os.path.join(output_directory, 'enhanced_{}'.format(filename)), 
            sample_rate,
            generated_audio[0].squeeze().cpu().numpy())
