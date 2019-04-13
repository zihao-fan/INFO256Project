import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_embeddings(filename, max_vocab_size):

    vocab={}
    embeddings=[]
    with open(filename) as file:
        
        cols=file.readline().split(" ")
        num_words=int(cols[0])
        size=int(cols[1])
        embeddings.append(np.zeros(size))  # 0 = 0 padding if needed
        embeddings.append(np.zeros(size))  # 1 = UNK
        vocab["_0_"]=0
        vocab["_UNK_"]=1
        
        for idx,line in enumerate(file):

            if idx+2 >= max_vocab_size:
                break

            cols=line.rstrip().split(" ")
            val=np.array(cols[1:])
            word=cols[0]
            
            embeddings.append(val)
            vocab[word]=idx+2

    return np.array(embeddings), vocab, size

def get_word_ids(docs, vocab, max_length=1000):
    
    doc_ids=[] 
    
    for doc in docs:
        wids=[]

        for token in doc[:max_length]:
            val = vocab[token.lower()] if token.lower() in vocab else 1
            wids.append(val)
        
        # pad each document to constant width
        for i in range(len(wids),max_length):
            wids.append(0)

        doc_ids.append(wids)

    return np.array(doc_ids)

def load_abstract_to_label(data_path, embeddings, vocab):
    data = np.load(data_path).item()
    X_train, y_train, X_val, y_val, X_test, y_test = data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']
    X_train = get_word_ids(X_train, vocab)
    X_val = get_word_ids(X_val, vocab)
    X_test = get_word_ids(X_test, vocab)
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    embeddings, vocab, embedding_size=load_embeddings(embedding_path, 100000)
    load_abstract_to_label('../data/dataset_abstract_stat_50.npy', embeddings, vocab)