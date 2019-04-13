import keras
import numpy as np
import json

from data_helper import load_embeddings,  load_abstract_to_label
import models

def train_and_val(model_path, 
                model, 
                X_train, 
                y_train, 
                X_val, 
                y_val, 
                batch_size=128, 
                epochs=10):
    model.fit(X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size)
    model.save(model_path)

if __name__ == '__main__':
    label2idx = json.load(open('../data/journal2idx.json', 'r'))
    label_num = len(label2idx) + 1
    embedding_path = '../data/glove.42B.300d.50K.w2v.txt'
    embeddings, vocab, embedding_size=load_embeddings(embedding_path, 100000)
    X_train, y_train, X_val, y_val, X_test, y_test = load_abstract_to_label('../data/dataset_abstract_stat_50.npy', embeddings, vocab)

    model = models.baseline_abstract_cnn_model(embeddings, len(vocab), embedding_size, label_num)
    train_and_val('../models/abstract_cnn_baseline', model, X_train, y_train, X_val, y_val)