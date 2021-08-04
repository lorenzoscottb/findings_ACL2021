
import argparse
import os
from utils import *
from datasets import * 

parser = argparse.ArgumentParser(description='PyTorch Embeddings Models')
# parser.add_argument('-model', type=str, default='dmsgns',
#                     help='model to use') # original file had other distributinal models (e.g. SGNS, GloVe), not included in this repo
parser.add_argument('-data', type=str, default='txt8',
                    help='corpus to use')
parser.add_argument('-data_type', type=str, default='single_str',
                    help='how the data is stored [single string/list of string]')
parser.add_argument('-save_folder', type=str, default='/mnt/data2/lb540/models/embeddings_models/embeddings/',
                    help='path to save the word vectors')
parser.add_argument('-save_file', type=bool, default=True,
                    help='path to save the word vectors')
parser.add_argument('-emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('-epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('-batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('-window_size', type=int, default=5,
                    help='context window size')
parser.add_argument('-neg_num', type=int, default=5,
                    help='negative samples per training example')
parser.add_argument('-min_count', type=int, default=5,
                    help='number of word occurrences for it to be included in the vocabulary')
parser.add_argument('-lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('-reduce_lr', type=bool, default=False,
                    help='lenearly reduce lr at each step')
parser.add_argument('-wd', type=float, default=0,
                    help='weight decay value. [1.0 L2 reg w/ Adam]')
parser.add_argument('-optimizer', type=str, default='adam',
                    help='optimizert to use [fixed lr!!!]')
parser.add_argument('-gpu', default='0',
                    help='GPU to use')
parser.add_argument('-eval_model', type=bool, default=False,
                    help='Gevaluate or not the model after training')
parser.add_argument('-norm', type=bool, default=False,
                    help='Normalize parameters during training')
parser.add_argument('-q', type=int, default=5,
                    help='number of negative values for ossgns matrix')
parser.add_argument('-epsilon', type=float, default=0.01,
                    help='epsilon to be used in the LogitSGNS model')
args = parser.parse_args()

pick_model = "dmsgns"
batch_size = args.batch_size
neg_sem = args.neg_num 
ws = args.window_size
min_c = args.min_count
vec_size = args.emsize
epochs = args.epochs
eval_model = args.eval_model
save_model = args.save_file
fldr = args.save_folder
use_norm = args.norm
lr = args.lr
wght_dc = args.wd
ds_name = args.data
mininal_lr = 0.00001 # as in the original word2vec C implementation, in case of decreasing gradiant


# Prepare dataset  
txt8_dir = '/path/to/dataset/in/text/form.txt' 
dtst = {'txt8':txt8_dir, 'clnwk':cln_wk_dir}  

dmsk_t8 = '/path/to/dataset/in/compressed/dictionary/form.pkl' 
dmsk = {'txt8':dmsk_t8, 'clnwk':dmsk_cl}

my_data = DataReader(dtst[args.data], min_count=min_c, ratio=1.0, fl_type=args.data_type)
my_data.compute_neg_sample_tensor() # collect negative sample tensor, as in the original word2vec C implementation
dataset = DMDataset(my_data, fl_type=args.data_type, neg_num=neg_sem, load_data=dmsk[ds_name])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=dataset.collate)
vocab_size = len(my_data.word2id)
epoch_size = dataset.data_len // batch_size # gives % of epochs
training_size = dataset.data_len * epochs # for lenear decreasing


# Prepare model and optimizer

skip_gram_model = models.DMSkipGramModel(vocab_size, vec_size, len(dataset.dep2id), neg_sem)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda: 
    print('Using GPU:'+str(device))
    skip_gram_model.cuda()
else:
    print(device)

if args.optimizer == 'adam':
    print('using Adam, bs:',args.batch_size,'lr', args.lr,'decrease lr:', args.reduce_lr)
    optimizer = torch.optim.Adam(skip_gram_model.parameters(), lr=lr, weight_decay=wght_dc)
elif args.optimizer == 'sgd':
    print('using SGD, bs:',args.batch_size,'lr', args.lr,'decrease lr:', args.reduce_lr)
    optimizer = torch.optim.SGD(skip_gram_model.parameters(), lr=lr)
else: 
    print("No such optimizer:", args.optimizer)
    exit(1)


# Training
done_samples = 0
for epoch in range(epochs):
    last_time = time.time()
    last_words = 0
    running_loss = 0.0
    
    for step, batch in enumerate(dataloader):
        pos_u = batch[0].to(device)
        pos_v = batch[1].to(device)
        neg_v = batch[2].to(device)
            
        optimizer.zero_grad()
        loss = skip_gram_model.forward(pos_u, pos_v, neg_v)
        loss.backward()
        optimizer.step()
        
        if args.reduce_lr:
            done_samples += len(batch[0])
            if optimizer.param_groups[0]['lr'] < args.lr * 0.0001:
                optimizer.param_groups[0]['lr'] = args.lr * 0.0001
            else:
                optimizer.param_groups[0]['lr'] = args.lr * (1-(done_samples/training_size))

        running_loss = running_loss * 0.9 + loss.item() * 0.1
        if step % (epoch_size // 10) == 10:                
            now_time = time.time()
            now_words = step * batch_size
            wps = (now_words - last_words) / (now_time - last_time)
            print('%.2f' % (step * 1.0 / epoch_size), end=' ')
            print('loss = %.3f' % running_loss, end=', ')
            print('wps = ' + str(int(wps)))

            last_time = now_time
            last_words = now_words

    print("Epoch: " + str(epoch + 1), end=", ")
    print("Loss = " + str(running_loss), end="\n")

