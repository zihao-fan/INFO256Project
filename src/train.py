import keras
import numpy as np
import json
import pickle
from keras.models import load_model

from data_helper import load_embeddings, load_abstract_to_label, load_ref_chain_to_label, load_ref_nb_to_label
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
    return model

def load_and_train_abstract():
    label2idx = json.load(open('../data/journal2idx.json', 'r'))
    label_num = len(label2idx) + 1
    embedding_path = '../data/glove.42B.300d.50K.w2v.txt'
    embeddings, vocab, embedding_size=load_embeddings(embedding_path, 100000)
    X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = load_abstract_to_label('../data/dataset_abstract_stat_50.npy', embeddings, vocab)
    model = models.baseline_abstract_cnn_model(embeddings, len(vocab), embedding_size, label_num)
    train_and_val('../models/abstract_cnn_baseline', model, X_train, y_train, X_val, y_val)

def load_and_train_ref_chain(has_rank=False):
    embedding_size = 50
    label2idx = json.load(open('../data/journal2idx.json', 'r'))
    label_num = len(label2idx) + 1
    journal2idx_all = json.load(open('../data/journal2idx_all.json', 'r'))
    X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = load_ref_chain_to_label('../data/dataset_ref_chain_stat_50.npy')
    print('Input journals', len(journal2idx_all), 'embedding_size', embedding_size, 'label_num', label_num)
    
    model = models.baseline_reference_chain_cnn_model(len(journal2idx_all), embedding_size, 
                                                    label_num, has_rank=has_rank)
    if has_rank:
        y_train = [np.array(y_train), np.array(y_train_r)]
        y_val = [np.array(y_val), np.array(y_val_r)]
    if has_rank:
        train_and_val('../models/ref_chain_cnn_with_rank', model, X_train, y_train, X_val, y_val)
    else:
        train_and_val('../models/ref_chain_cnn_baseline', model, X_train, y_train, X_val, y_val)

def load_and_train_ref_neighbor(has_rank=False):
    embedding_size = 50
    label2idx = json.load(open('../data/journal2idx.json', 'r'))
    label_num = len(label2idx) + 1
    journal2idx_all = json.load(open('../data/journal2idx_all.json', 'r'))
    X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = load_ref_nb_to_label('../data/dataset_ref_nb_stat_50.npy')
    print('Input journals', len(journal2idx_all), 'embedding_size', embedding_size, 'label_num', label_num)
    model = models.baseline_reference_neighbors_cnn_model(len(journal2idx_all), embedding_size, 
                                                        label_num, has_rank=has_rank)
    if has_rank:
        y_train = [np.array(y_train), np.array(y_train_r)]
        y_val = [np.array(y_val), np.array(y_val_r)]
    if has_rank:
        train_and_val('../models/ref_neighbor_cnn_with_rank', model, X_train, y_train, X_val, y_val)
    else:
        train_and_val('../models/ref_neighbor_cnn_baseline', model, X_train, y_train, X_val, y_val)

def load_and_train_ref_abs(has_rank=False):
    label2idx = json.load(open('../data/journal2idx.json', 'r'))
    label_num = len(label2idx) + 1
    embedding_path = '../data/glove.42B.300d.50K.w2v.txt'
    embeddings, vocab, embedding_size=load_embeddings(embedding_path, 100000)
    X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = load_abstract_to_label('../data/dataset_abstract_stat_50.npy', 
                                                                        embeddings, vocab)

    label2idx = json.load(open('../data/journal2idx.json', 'r'))
    label_num = len(label2idx) + 1
    journal2idx_all = json.load(open('../data/journal2idx_all.json', 'r'))

    X_train_ref, _,  _, X_val_ref, _, y_val_r_ref, X_test_ref, _, _ = load_ref_chain_to_label('../data/dataset_ref_chain_stat_50.npy')

    model = models.reference_abstract_model(embeddings, 
                    word_vocab_size=len(vocab), 
                    word_embedding_dim=embedding_size, 
                    ref_vocab_size=len(journal2idx_all), 
                    ref_embedding_dim=embedding_size, 
                    label_num=label_num,
                    has_rank=has_rank)

    if has_rank:
        y_train = [np.array(y_train), np.array(y_train_r)]
        y_val = [np.array(y_val), np.array(y_val_r)]
        train_and_val('../models/ref_abs_cnn_with_rank', model, [np.array(X_train), np.array(X_train_ref)], 
                        y_train, [np.array(X_val), np.array(X_val_ref)], y_val)
    else:
        train_and_val('../models/ref_abs_cnn_baseline', model, [np.array(X_train), np.array(X_train_ref)], 
                        y_train, [np.array(X_val), np.array(X_val_ref)], y_val)


def produce_test_prediciton(model_path, X_test, save_name):
    model = load_model(model_path)
    prediction = model.predict(X_test)
    # print(prediction.shape)
    with open(save_name, 'wb') as f:
        pickle.dump(prediction, f)

if __name__ == '__main__':
    # load_and_train_abstract()
    # load_and_train_ref_chain(has_rank=True)
    # load_and_train_ref_neighbor(has_rank=True)
    load_and_train_ref_abs(has_rank=True)

    # embedding_path = '../data/glove.42B.300d.50K.w2v.txt'
    # embeddings, vocab, embedding_size=load_embeddings(embedding_path, 100000)
    # X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = load_abstract_to_label('../data/dataset_abstract_stat_50.npy', embeddings, vocab)
    # produce_test_prediciton('../models/abstract_cnn_baseline', X_test, '../outputs/abstract_cnn_baseline')

    # X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = load_ref_chain_to_label('../data/dataset_ref_chain_stat_50.npy')
    # produce_test_prediciton('../models/ref_chain_cnn_baseline', X_test, '../outputs/ref_chain_cnn_baseline')
    # produce_test_prediciton('../models/ref_chain_cnn_with_rank', X_test, '../outputs/ref_chain_cnn_with_rank')

    # X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = load_ref_nb_to_label('../data/dataset_ref_nb_stat_50.npy')
    # produce_test_prediciton('../models/ref_neighbor_cnn_baseline', X_test, '../outputs/ref_neighbor_cnn_baseline')
    # produce_test_prediciton('../models/ref_neighbor_cnn_with_rank', X_test, '../outputs/ref_neighbor_cnn_with_rank')

    embedding_path = '../data/glove.42B.300d.50K.w2v.txt'
    embeddings, vocab, embedding_size=load_embeddings(embedding_path, 100000)
    X_train, y_train, y_train_r, X_val, y_val, y_val_r, X_test, y_test, y_test_r = load_abstract_to_label('../data/dataset_abstract_stat_50.npy', 
                                                                        embeddings, vocab)

    label2idx = json.load(open('../data/journal2idx.json', 'r'))
    label_num = len(label2idx) + 1
    journal2idx_all = json.load(open('../data/journal2idx_all.json', 'r'))

    X_train_ref, _, _, X_val_ref, _, _, X_test_ref, _, _ = load_ref_chain_to_label('../data/dataset_ref_chain_stat_50.npy')
    produce_test_prediciton('../models/ref_abs_cnn_with_rank', 
                    [np.array(X_test), np.array(X_test_ref)],
                    '../outputs/ref_abs_with_rank')