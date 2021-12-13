from sent2vec.vectorizer import Vectorizer
import pdb 

sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]


vectorizer = Vectorizer()

pdb.set_trace()
vectorizer.bert(sentences)
vectors = vectorizer.vectors

print(vectors)