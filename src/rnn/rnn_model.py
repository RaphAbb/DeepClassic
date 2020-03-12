'''
DeeProject: p Classic (2019)
rnn_model.py: RNN model for music generation
Author: Raphael Abbou
'''
import os

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

import sys

class SimpleRNN(nn.Module):
    ''' Simple RNN
    '''
    #
    def __init__(self, emb_size, hidden_size, dropout = 0.6):
        super(SimpleRNN, self).__init__()
        
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.rnn = nn.RNN(self.emb_size, self.hidden_size, nonlinearity = 'tanh', bias = True, dropout = dropout)
        
        self.decoder = nn.Linear(self.hidden_size, self.emb_size, bias = False) #bias = False by similarity with NMT
        
    
    def forward(self, inputs, hidden):
        outputs, last_hidden = self.rnn.forward(inputs , hidden)
        
        outputs = self.decoder(outputs)
        
        return outputs, last_hidden

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        length, batch_size = char_sequence.size()
        input = char_sequence[:length-1,:] #remove <END> token for each elem in the batch
        target = char_sequence[1:,:] #remove <START> token
        
        scores, (last_hidden, last_cell) = self.forward(input, dec_hidden)
        
        logits = scores.contiguous().view((length-1)*batch_size, len(self.target_vocab.char2id)) # Flattened vector of ouput char distribution
        y_train = target.contiguous().view((length-1)*batch_size) #Flattened vector of input char indices
        
        loss_func = nn.CrossEntropyLoss(reduction="sum", ignore_index = self.target_vocab.char_pad)
        loss = loss_func(logits, y_train)
        
        return loss
        ### END YOUR CODE


    @staticmethod
    def load(model_path: str, no_char_decoder=False):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = SimpleRNN(**args)
        #NMT(vocab=params['vocab'], no_char_decoder=no_char_decoder, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(emb_size = self.emb_size, hidden_size = self.hidden_size, dropout = self.dropout),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
        
if __name__ == "__main__":
    os.chdir('../../')
    from src.numpy_encode import *
    from src.config import *
    from src.music_transformer import *
    from src.utils.midifile import *
    from src.utils.file_processing import process_all
    
    from src.utils import midifile
    
    midi_file = Path("maestro-v2.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.midi")
    vocab = MusicVocab.create()
    item = MusicItem.from_file(midi_file, vocab)
    
    #input_tensor = item.to_tensor()
    #We try to keep the 1-hot encoding
    mf = midifile.file2mf(midi_file)
    stream = file2stream(mf)
    chordarr = stream2chordarr(stream) #(timestep, nb_tracks, embedding_size)
    chordarr = chordarr.squeeze(1) #We only have one track == piano track
    
    #inputs = input_tensor.unsqueeze(0)
    inputs = torch.Tensor(chordarr)
    inputs = inputs.unsqueeze(1) #Tensor: (timestep, batch_size, embedding_size)

    hidden_size = 10
    emb_size = chordarr.shape[-1]
    dropout = 0.7
    
    #inputs = inputs.permute(0,2,1) # Tensor: (batch_size, emb_size, timesteps)
    
    hidden = torch.zeros(1, 1, hidden_size)
   
    rnn = nn.RNN(emb_size, hidden_size, nonlinearity = 'tanh', bias = True)
    output, hidden = rnn.forward(inputs , hidden)
    probas = F.softmax(output, -1)
    #probas.sum(-1) #-> verifies that softmax along the right dim
    
    
    
    
    
    