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

def get_rows_with_same_id(row): 
    return row.split("_")[0]

def group_perplexity(group, perplexity): 
    pdb.set_trace()



def extract_features(path_to_df, use_rank): 
    df = pd.read_csv(path_to_df, sep='\t')
    
    # drop the unnamed column 
    df = df.drop(columns=['Unnamed: 0'])

    if use_rank: 
        df['group'] = df['ids'].apply(lambda x: get_rows_with_same_id(x))
        # ranked score 

        # get the group ids starting with 1, ending with the length of the df. 
        dataframe_with_ranks_df = {"ids": [], "rank":[]}
        for i in range(1,(len(df)//5)+1): 
            subset = df.loc[df['group'] == str(i)]

            d = []
            for ids, perplexity in zip(subset['ids'].tolist(), subset['perplexity'].tolist()): 
                d.append([ids, perplexity])

            # sort the tuple: lower perplexity is on the first position. 
            sorted_d = sorted(d, key=lambda tup: tup[1])
            
            # assign rank 
        
            for index, elem in enumerate(sorted_d,1):
                dataframe_with_ranks_df["ids"].append(elem[0])
                dataframe_with_ranks_df["rank"].append(index)
            
        dataframe_with_ranks = pd.DataFrame.from_dict(dataframe_with_ranks_df)
        merged_df = pd.merge(df, dataframe_with_ranks, on='ids')
        merged_df = merged_df.drop(columns=['fillers', 'perplexity', 'group', 'context_before', 'context_after', 'sents_with_filler'])
        path_to_new_df = "./data/{0}_feat.csv".format(path_to_df.replace('.tsv', ''))
        merged_df.to_csv(path_to_new_df)
        print(merged_df)
    else: 
        df.drop(columns=['fillers', 'context_before', 'context_after', 'sents_with_filler'])
        path_to_new_df = "./data/{0}_feat_perplexity.csv".format(path_to_df.replace('.tsv', ''))
        df.to_csv(path_to_new_df)
    return path_to_new_df

