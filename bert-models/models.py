from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch 
from torch.autograd import Variable
import pdb

class BERTClassification(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout, LSTM=True):
        
        super().__init__()
        
        self.bert = bert
        self.lstm = LSTM 
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        embedding_dim = bert.config.to_dict()['hidden_size']
        
        if self.lstm: 
            self.rnn = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = dropout) 
        else: 
            self.rnn = nn.GRU(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            batch_first = True,
                            dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]

        if self.lstm: 
            h_0 = Variable(torch.randn(self.n_layers*2, embedded.size()[0], self.hidden_dim)).cuda()
            c_0 = Variable(torch.randn(self.n_layers*2, embedded.size()[0], self.hidden_dim)).cuda()
            _, hidden = self.rnn(embedded, (h_0, c_0)) 
            hidden = self.dropout(torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim = 1))

        else:   
            _, hidden = self.rnn(embedded)
            
            #hidden = [n layers * n directions, batch size, emb dim]
            
            if self.rnn.bidirectional:
                hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            else:
                hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output
