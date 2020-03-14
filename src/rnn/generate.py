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

from run_rnn import filter_indexes, predict


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

    