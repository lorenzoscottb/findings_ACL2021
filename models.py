
__author__ = 'lb540'

import torch 
import numpy as np
from torch import FloatTensor as FT

class DMSkipGramModel(torch.nn.Module):

    def __init__(self, vocab_size, emb_dimension, num_dep, neg_smp, clmpv=6, weights=None):
        print('Initialising DM model...')
        super(DMSkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
        self.v_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
        self.clmpv = clmpv
        self.neg_smp = neg_smp
        self.num_dep = num_dep
        initrange = 1.0 / self.emb_dimension
        torch.nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        torch.nn.init.constant_(self.v_embeddings.weight.data, 0)     

        self.dep_mxs = torch.nn.Embedding(self.num_dep, self.emb_dimension*self.emb_dimension)
        
        # starting matrices as identity matrices
        self.dep_mxs.weight = torch.nn.Parameter(torch.eye((self.emb_dimension)).view(1, self.emb_dimension*self.emb_dimension).repeat(1,self.num_dep).view(self.num_dep, self.emb_dimension*self.emb_dimension))
        self.dep_mxs.weight.requires_grad = True
        
        self.weights = weights
        if self.weights is not None:
            assert min(self.weights) >= 0, "Each weight should be >= 0"
            self.weights = torch.autograd.Variable(torch.from_numpy(weights)).float()
            
    def sample(self, size_to_sample):
        return torch.multinomial(self.weights, size_to_sample, True)

    def forward(self, input_label, out_label, dep_label, use_given=None):

        batch_size = out_label.size()[0]
        input_word = self.u_embeddings(input_label)
        output_word = self.v_embeddings(out_label)

        dep_vec = self.dep_mxs(dep_label)
        dep_mx = dep_vec.view(batch_size, self.emb_dimension, self.emb_dimension)
        
        if self.weights is not None:
            noise = self.weights[self.sample(self.neg_smp*batch_size)].long()    
        elif use_given is not None:
            noise = use_given
            num_sampled = len(use_given[0])
        else:
            noise = torch.autograd.Variable(torch.Tensor(batch_size, self.neg_smp).
                             uniform_(0, self.vocab_size - 1).long())
        
        if self.u_embeddings.weight.is_cuda:
            noise = noise.cuda()
        noise = self.v_embeddings(noise).neg()
        
        output_word = output_word.unsqueeze(2)
        output_word = torch.matmul(dep_mx, output_word).view(batch_size, self.emb_dimension)
        vec_dot = (input_word * output_word).sum(1).squeeze()        
        log_target = vec_dot.sigmoid().log()     

        noise = (dep_mx.unsqueeze(1) @ noise.unsqueeze(-1)).view(batch_size, self.neg_smp, self.emb_dimension)    
        sum_sampled = torch.bmm(noise, input_word.unsqueeze(2))
        sum_log_sampled = sum_sampled.sigmoid().log().sum(1).squeeze()
        loss = log_target + sum_log_sampled

        return -loss.sum() / batch_size
