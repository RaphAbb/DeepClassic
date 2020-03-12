# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:14:52 2020

@author: rapha
"""
import glob
import os 

def folder2scores(path, vocab):
    scores = []
    print("Processing file: ")
    for file in glob.glob(path + "/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO*.midi"):
        print(file)
        midi_file = Path(file)
        item = MusicItem.from_file(midi_file, vocab)
        scores.append(item.to_tensor()) # Tensor: (timesteps)
        
        
    return scores
        
        
if __name__ == "__main__":
    
    os.chdir('../../')

    from src.numpy_encode import *
    from src.config import *
    from src.music_transformer import *
    from src.utils.midifile import *
    from src.utils.file_processing import process_all
    
    from src.utils import midifile
    
    from src.rnn.rnn_model import SimpleRNN

    from src.rnn.run_rnn import train_for_epoch, train, tensor2onehot, batch_tensor2onehot, pad_emb_scores
    
    path = "maestro-v2.0.0/2015"
    
    vocab = MusicVocab.create()

    scores = folder2scores(path, vocab)
    inputs = pad_emb_scores(scores, vocab)
    
    hidden_size = 512
    emb_size = len(vocab)
    dropout = 0.7
    
    hidden = torch.zeros(1, 1, hidden_size)
    
    rnn = SimpleRNN(emb_size, hidden_size)

    optimizer = optim.Adam(rnn.parameters())
    loss_func = nn.CrossEntropyLoss(reduction='mean')

    output_dir = "src/rnn/results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train(rnn, train_data=scores, vocab=vocab, output_path=output_path, batch_size=64, n_epochs=10, lr=0.0005)