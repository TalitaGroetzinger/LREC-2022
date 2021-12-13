import torch
import torch.nn as nn
from torch.autograd import Variable


class BERTClassification(nn.Module):
    def __init__(
        self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout, num_features=1, LSTM=True
    ):
        super().__init__()

        self.bert = bert
        self.lstm = LSTM
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_features = num_features

        embedding_dim = bert.config.to_dict()["hidden_size"]

        if self.lstm:
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout,
            )
        else:
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=0 if n_layers < 2 else dropout,
            )

        self.out = nn.Linear((hidden_dim * 2)+self.num_features, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, rank):
        # text = [batch size, sent len]
        embedded = self.bert(text)[0]
        # embedded = [batch size, sent len, emb dim]

        if self.lstm:
            if torch.cuda.is_available():
                h_0 = Variable(torch.randn(self.n_layers * 2, embedded.size()[0], self.hidden_dim)).cuda()
                c_0 = Variable(torch.randn(self.n_layers * 2, embedded.size()[0], self.hidden_dim)).cuda()
            else:
                h_0 = Variable(torch.randn(self.n_layers * 2, embedded.size()[0], self.hidden_dim))
                c_0 = Variable(torch.randn(self.n_layers * 2, embedded.size()[0], self.hidden_dim))

            _, hidden = self.rnn(embedded, (h_0, c_0))
            hidden = self.dropout(
                torch.cat((hidden[0][-2, :, :], hidden[0][-1, :, :]), dim=1)
            )

        else:
            _, hidden = self.rnn(embedded)
            # hidden = [n layers * n directions, batch size, emb dim]

            if self.rnn.bidirectional:
                hidden = self.dropout(
                    torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                )
            else:
                hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        ranking_var = rank.unsqueeze(1)
        final_rep = torch.cat([hidden, ranking_var], 1) 
        output = self.out(final_rep)

        # output = [batch size, out dim]

        return output



class SimpleBERT(nn.Module):
    def __init__(self,
                 bert,
                 output_dim): 
        
        super().__init__()
        
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.output_dim = output_dim
        

        
        self.out = nn.Linear(embedding_dim, output_dim)

        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        embedded = self.bert(text)[1]
                
        #embedded = [batch size, sent len, emb dim]

     
        #hidden = [batch size, hid dim]
 
        output = self.out(embedded)
        
        
        #output = [batch size, out dim]
        
        return output