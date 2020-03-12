#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeeProject: p Classic (2019)
generate.py: 
Author: Raphael Abbou
"""
from datetime import datetime
import os
import pickle
import math
import time
import argparse
import numpy as np

from torch import nn, optim
import torch
from tqdm import tqdm

import os

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from rnn_model import SimpleRNN
from run_rnn import tensor2onehot


def filter_indexes(res, prev_idx, vocab, filter_value=-float('Inf')):
    res[vocab.stoi['xxpad']] = filter_value
    res[vocab.dur_range[0]+17:] = filter_value
    
    if vocab.is_duration_or_pad(prev_idx):
        res[list(range(*vocab.dur_range))] = filter_value
    elif prev_idx == vocab.stoi['xxsep']:
        res[list(range(*vocab.note_range))] = filter_value
        res[vocab.stoi['xxsep']] = filter_value
    elif prev_idx == vocab.stoi['xxbos']:
        res[list(range(*vocab.dur_range))] = filter_value
        res[vocab.stoi['xxsep']] = filter_value
    else:
        res[list(range(*vocab.note_range))] = filter_value
    
    return res

def predict(model, vocab, T):
    x_t = tensor2onehot(np.array([vocab.stoi['xxbos']]), vocab).unsqueeze(1)
    h_t = torch.zeros(1, 1, model.hidden_size)
    
    output = [vocab.stoi['xxbos']]
    prev_idx = vocab.stoi['xxbos']
    
    t = 0
    while t <= T:
        xout_t, h_t = model.forward(x_t, h_t)
        xout_t_corrected = filter_indexes(xout_t.squeeze(), prev_idx, vocab)
        xhat_t = F.softmax(xout_t_corrected)
        
        x_t = np.random.choice(model.emb_size, 1, p=xhat_t.detach().numpy())
        #while x_t[0] == vocab.stoi['xxpad']:
            #while vocab.is_duration_or_pad(x_t[0]) == is_duration_prev:
                #vocab.dur_range --> look at music_transformer/learner.py
        prev_idx = x_t[0]
        output.append(x_t[0])
        x_t = tensor2onehot(x_t, vocab).unsqueeze(1)
        
        t+=1

    output.append(vocab.stoi['xxeos'])
    
    return output


if __name__ == "__main__":
    
    os.chdir('../../')
    from src.numpy_encode import *
    from src.config import *
    from src.music_transformer import *
    from src.utils.midifile import *
    from src.utils.file_processing import process_all
    
    from src.utils import midifile
    
    #model_path = "src/rnn/results/20200306_181925/model.weights"
    model_path = "src/rnn/results/20200306_220802/model.weights"
    my_rnn = SimpleRNN.load(model_path)
    
    vocab = MusicVocab.create()
    
    output = predict(my_rnn, vocab, 16*20)
    print(vocab.textify(output))
    
    pred = vocab.to_music_item(np.array(output))
    pred.show()


## Colab cannot play music directly from music21 - must convert to .wav first
#def play_wav(stream):
#    out_midi = stream.write('midi')
#    out_wav = str(Path(out_midi).with_suffix('.wav'))
#    FluidSynth("font.sf2").midi_to_audio(out_midi, out_wav)
#    return Audio(out_wav)
#
#
#    from midi2audio import FluidSynth
#    from IPython.display import Audio
#    pred.show()
#    play_wav(pred.stream)

    