import torch 
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import math 
import numpy as np 


class GPTScorer: 

    def __init__(self, tokenizer, model, sequence="Hello my dog is cute"): 
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.sequence = sequence
        self.indices = self.tokenizer.encode(self.sequence)
        self.repr_in_model = self.tokenizer.tokenize(self.sequence)

    
    def get_logits(self): 
        inputs = self.tokenizer(self.sequence, return_tensors='pt')
        print("indices")

        # alternative: eliminate encode and use indices["input_ids"]
        indices = self.tokenizer.encode(self.sequence)
        decoded_indices = self.tokenizer.decode(indices)
        # returns the loss and logits 
        # loss:  (torch.FloatTensor of shape (1,): Language modeling loss (for next-token prediction).
        # logits (1): (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) 
        # logits (2): Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        # size of the logits is torch.Size([1, 7, 40478]) -> 1, seq_len, vocab_size 
        logits = outputs.logits
        print("logits")
        print(torch.logsumexp(logits.squeeze(0), 1))
        softmax_logits = torch.softmax(logits.squeeze(0),1)

        #print(np.sum([prob for prob in softmax_logits[2].tolist()]))
        scores_for_each_token = []
        scores = 0 
        for i, index in enumerate(self.indices,0): 
            individual_score_for_token = softmax_logits[i][index].item()
            print(math.log(individual_score_for_token))
            scores += individual_score_for_token
            scores_for_each_token.append(math.log(individual_score_for_token)) 
        #scores_in_log_space = np.log(scores_for_each_token).tolist()
        #print(scores_in_log_space)
        final_score = np.prod(scores_for_each_token)
        print(scores)
        return final_score

    def get_perplexity(self): 
        """
            Based on : https://github.com/huggingface/transformers/issues/473 (Thomas Wolf approved XD)
        """
        inputs = self.tokenizer(self.sequence)
        indices = torch.tensor([self.tokenizer.encode(self.sequence)])
        # tensor([[ 655,  544,  246, 1861,  504,  481, 2181]])
        decoded_indices = self.tokenizer.decode(inputs["input_ids"])
        outputs = self.model(indices, labels=indices)
        #print(outputs)
        loss = outputs.loss.item()
        return math.exp(loss)
        



if __name__ == "__main__":
    scorer = GPTScorer("There is a book on the desk")
    perplexity1 = scorer.get_perplexity()
    prob1 = scorer.get_logits()
    print(perplexity1)
    print(prob1)

    scorer = GPTScorer("There is a plane on the desk")
    perplexity2 = scorer.get_perplexity()
    prob2 = scorer.get_logits()
    print(perplexity2)
    print(prob2)


    if prob2 < prob1: 
        print("plane is smaller than book")
    else: 
        print("book is smaller than plane")