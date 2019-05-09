import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_embeddings(filename, max_vocab_size):

    vocab={}
    embeddings=[]
    with open(filename, 'r', encoding="utf-8") as file:
        
        cols=file.readline().split(" ")
        num_words=int(cols[0])
        size=int(cols[1])
        embeddings.append(np.zeros(size))  # 0 = 0 padding if needed
        embeddings.append(np.zeros(size))  # 1 = UNK
        vocab["_PAD_"]=0
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

def pad_ref(refs, max_length = 200):
    
    ref_ids = []
    for ref in refs:
        rids = []
        for item in ref[:max_length]:
            rids.append(item)
        for i in range(len(rids), max_length):
            rids.append(0)
        ref_ids.append(rids)
    return np.array(ref_ids)

def load_abstract_to_label(data_path, embeddings, vocab):
    data = np.load(data_path).item()
    X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = data['X_train_abstract'], data['y_train_journal'], data['y_train_rank'],\
                                                    data['X_val_abstract'], data['y_val_journal'], data['y_val_rank'],\
                                                    data['X_test_abstract'], data['y_test_journal'], data['y_test_rank']
    X_train = get_word_ids(X_train, vocab)
    X_val = get_word_ids(X_val, vocab)
    X_test = get_word_ids(X_test, vocab)
    return X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r

def load_ref_chain_to_label(data_path):
    data = np.load(data_path).item()
    X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = data['X_train_ref_chain'], data['y_train_journal'], data['y_train_rank'],\
                                        data['X_val_ref_chain'], data['y_val_journal'], data['y_val_rank'], \
                                        data['X_test_ref_chain'], data['y_test_journal'], data['y_test_rank']
    X_train = pad_ref(X_train)
    X_val = pad_ref(X_val)
    X_test = pad_ref(X_test)
    return X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r

def load_ref_nb_to_label(data_path):
    data = np.load(data_path).item()
    X_train_1, X_train_2, y_train, y_train_r, X_val_1, X_val_2, y_val, y_val_r, X_test_1, X_test_2, y_test, y_test_r = data['X_train_ref_level1'],data['X_train_ref_level2'], data['y_train_journal'], data['y_train_rank'],\
                                        data['X_val_ref_level1'], data['X_val_ref_level2'], data['y_val_journal'], data['y_val_rank'], \
                                        data['X_test_ref_level1'], data['X_test_ref_level2'], data['y_test_journal'], data['y_test_rank']
    
    data = X_train_1, X_train_2, y_train, y_train_r, X_val_1, X_val_2, y_val, y_val_r, X_test_1, X_test_2, y_test, y_test_r
    data = tuple([np.array(item) for item in data])
    X_train_1, X_train_2, y_train, y_train_r, X_val_1, X_val_2, y_val, y_val_r, X_test_1, X_test_2, y_test, y_test_r = data
    X_train = [X_train_1, X_train_2]
    X_val = [X_val_1, X_val_2]
    X_test = [X_test_1, X_test_2]
    print(X_train_1[:10])
    print(X_train_2[:10])
    print(y_train[:10])
    print(y_train_r[:10])
    return X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r

if __name__ == '__main__':

    # embeddings, vocab, embedding_size=load_embeddings(embedding_path, 100000)
    # load_abstract_to_label('../data/dataset_abstract_stat_50.npy', embeddings, vocab)

    # load_ref_chain_to_label('../data/dataset_ref_chain_stat_50.npy')
    load_ref_nb_to_label('../data/dataset_ref_nb_stat_50.npy')