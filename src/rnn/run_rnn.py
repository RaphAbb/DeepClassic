#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeeProject: p Classic (2019)
run_rnn.py: training functions and utilities
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


#parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
#parser.add_argument('-d', '--debug', action='store_true', help='whether to enter debug mode')
#args = parser.parse_args()


def pad_emb_scores(scores, vocab):
    """ Pad list of music scores embeddings
    @param inputs (list[tensor]): list of embeddings for each score in batch
    @param vocab (Vocab)
    @returns padded_batch (list[list[int]]): tensor of scores where scores shorter
        than the max length scores are padded out with the pad_token, such that
        each score in the batch now has equal length.
        Output shape: (batch_size, max_score_length)
    """
    max_len = max(len(score) for score in scores)
    padded_batch = []

    for i in range(len(scores)):
        padded_batch.append(torch.cat((scores[i], torch.zeros(max_len-scores[i].shape[0]).long()+vocab.stoi['xxpad'])))
    
    padded_batch = torch.stack(padded_batch, dim = 0)

    return padded_batch



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tensor2onehot(tensor, vocab):
    """"Converts tensor to tensor of one-hot vectors"""
    one_hot = torch.zeros(tensor.shape[0], len(vocab))
    for i in range(tensor.shape[0]):
        one_hot[i][tensor[i]] = 1
    return one_hot

def batch_tensor2onehot(pad_scores, vocab):
    """Converts a tensor/list of padded tensors to one-hots"""
    batch_1hot = []
    for i in range(1, len(pad_scores)):
        batch_1hot.append(tensor2onehot(pad_scores[i], vocab))
    batch_1hot = torch.stack(batch_1hot, dim=1)
        
    return batch_1hot

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
        
# -----------------
# Primary Functions
# -----------------
def targetTensor(chordarr):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def train(rnn, train_data, vocab, output_path, batch_size=8, n_epochs=10, lr=0.0005):
    """ Train music RNN.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """

    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###      1) Construct Adam Optimizer in variable `optimizer`
    ###      2) Construct the Cross Entropy Loss Function in variable `loss_func` with `mean`
    ###         reduction (default)
    ###
    ### Hint: Use `parser.model.parameters()` to pass optimizer
    ###       necessary parameters to tune.
    ### Please see the following docs for support:
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    optimizer = optim.Adam(rnn.parameters())
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        curr_train_loss = train_for_epoch(rnn, train_data, vocab, optimizer, loss_func, batch_size)
        if True:
            print("Saving model. Loss:{}".format(curr_train_loss))
            rnn.save(output_path)
        print("")


def train_for_epoch(rnn, scores, vocab, optimizer, loss_func, batch_size):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param parser (RNN): Recurrent Neural Network
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size
    @param lr (float): learning rate

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    rnn.train() # Places model in "train" mode, i.e. apply dropout layer
    n_minibatches = math.ceil(len(scores) / batch_size)
    loss_meter = AverageMeter()
    
    pad_scores = pad_emb_scores(scores, vocab)
    onehot_inputs = batch_tensor2onehot(pad_scores, vocab)

    with tqdm(total=(n_minibatches)) as prog:
        optimizer.zero_grad()   # remove any baggage in the optimizer
        loss = 0. # store loss for this batch here
        T, B, E = onehot_inputs.shape # Timesteps, Batch_size, Embedding_size
        train_y = pad_scores[:,1:] #shift all by 1
        #train_y = inputs[1:,:,:] #shift all by 1
        #train_y = torch.cat((train_y, torch.zeros((1, B, E))), 0) #All 0's for the last time step
        
        #for x_train, y_train in batch_iter(data, batch_size, suffle=True):
        ### YOUR CODE HERE (~5-10 lines)
        ### Description
        ###      1) Run train_x forward through model to produce `logits`
        ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
        ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
        ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
        ###         are the predictions (y^ from the PDF).
        ###      3) Backprop losses
        ###      4) Take step with the optimizer
        ### Remarks:
        ###      - the loss does not take one-hot vectors as targets, but the correct class index
        hidden = torch.zeros(1, B, rnn.hidden_size)
        logits, last_hidden  = rnn.forward(onehot_inputs, hidden) # Tensor: (timesteps, batch_size, emb_size)
        loss = 0
        for b in range(B):
            #Rk: the loss does not take
            loss += loss_func(logits[:-1,b,:], train_y[b,:])/B
        loss.backward()
        optimizer.step()
        ### END YOUR CODE
        prog.update(1)
        loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))
    
    return loss.data.item()

    #print("Evaluating on dev set",)
    #parser.model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
    #dev_UAS, _ = parser.parse(dev_data)
    #print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    #return dev_UAS

def filter_invalid_indexes(res, prev_idx, vocab, filter_value=-float('Inf')):
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
    res[vocab.stoi['xxpad']] = filter_value
    
    return res

def predict(model, vocab, T):
    x_t = tensor2onehot(np.array([vocab.stoi['xxbos']]), vocab).unsqueeze(1)
    h_t = torch.zeros(1, 1, model.hidden_size)
    
    output = [vocab.stoi['xxbos']]
    prev_idx = vocab.stoi['xxbos']
    
    t = 0
    while t <= T:
        xout_t, h_t = model.forward(x_t, h_t)
        xout_t_corrected = filter_invalid_indexes(xout_t.squeeze(), prev_idx, vocab)
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
    
    from src.rnn.rnn_model import SimpleRNN


    vocab = MusicVocab.create()
    midi_file = Path("maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.midi")
    midi_file2 = Path("maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.midi")
    midi_file3 = Path("maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--4.midi")
    midi_file4 = Path("maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--5.midi")
    
    item1 = MusicItem.from_file(midi_file, vocab)
    item2 = MusicItem.from_file(midi_file2, vocab)
    item3 = MusicItem.from_file(midi_file3, vocab)
    item4 = MusicItem.from_file(midi_file4, vocab)
####Train Set
    inputs1 = item1.to_tensor() # (timesteps)
    inputs2 = item2.to_tensor() # (timesteps)
    scores = [inputs1, inputs2]
    inputs = pad_emb_scores(scores, vocab)

####Dev Set
    inputs3 = item3.to_tensor() # (timesteps)
    inputs4 = item4.to_tensor() # (timesteps)
    dev_scores = [inputs3, inputs4]
    dev_inputs = pad_emb_scores(dev_scores, vocab)
    
    hidden_size = 512
    emb_size = len(vocab)
    dropout = 0.7
    
    hidden = torch.zeros(1, 1, hidden_size)
    
    rnn = SimpleRNN(emb_size, hidden_size)

    optimizer = optim.Adam(rnn.parameters())
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    

    train_for_epoch(rnn, scores=inputs, vocab = vocab,
                    optimizer=optimizer, loss_func=loss_func, batch_size=1)

    output_dir = "src/rnn/results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train(rnn, train_data=scores, vocab=vocab, output_path=output_path, batch_size=64, n_epochs=10, lr=0.0005)