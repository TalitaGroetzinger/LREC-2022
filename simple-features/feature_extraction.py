import pandas as pd 
import pdb 
import numpy as np 
from perplexity import GPTScorer
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', return_dict=True).eval()


def filler_in_text(text_before, text_after,filler): 
    if filler in text_before or filler in text_after: 
       return 1 
    else: 
        return 0 

def get_perplexity(sentence_with_filler): 
    scorer = GPTScorer(tokenizer, model, sentence_with_filler)
    perplexity_score = scorer.get_perplexity()
    return perplexity_score 

def extract_features(df, filename): 
    # check if the word occures earlier in the text 
    #df['occurs_earlier'] = df.apply(lambda x: filler_in_text(x['context_before'], x['context_after'],x['fillers']), axis=1)
    print("compute perplexity")
    df['perplexity'] = df.apply(lambda x: get_perplexity(x['sents_with_filler']), axis=1) 

    df.to_csv(filename, sep='\t')


