os.chdir('../../')

from src.numpy_encode import *
from src.config import *
from src.music_transformer import *
from src.utils.midifile import *
from src.utils.file_processing import process_all

from src.utils import midifile

from src.rnn.rnn_model import SimpleRNN
from src.rnn.batch_load import folder2scores

from src.rnn.run_rnn import train_for_epoch, train, tensor2onehot, batch_tensor2onehot, pad_emb_scores

path = "maestro-v2.0.0"

years = ['2004', '2006', '2008', '2009'
         '2011', '2013', '2014', '2015', '2017', '2018']

vocab = MusicVocab.create()

scores = []
for Y in years:
    temp_path = path + "/" + Y
    scores = folder2scores(temp_path, vocab)

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