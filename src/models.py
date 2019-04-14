import keras
from keras.layers import Dense, Input, Embedding, GlobalMaxPooling1D, Conv1D, Concatenate, Dropout
from keras.models import Model, Sequential
from keras.utils.vis_utils import model_to_dot

def baseline_abstract_cnn_model(embeddings, vocab_size, word_embedding_dim, label_num):
    word_sequence_input = Input(shape=(None,), dtype='int32')
    
    word_embedding_layer = Embedding(vocab_size,
                                    word_embedding_dim,
                                    weights=[embeddings],
                                    trainable=False)

    
    embedded_sequences = word_embedding_layer(word_sequence_input)
    
    cnn2=Conv1D(filters=50, kernel_size=2, strides=1, padding="same", activation="tanh", name="CNN_bigram")(embedded_sequences)
    cnn3=Conv1D(filters=50, kernel_size=3, strides=1, padding="same", activation="tanh", name="CNN_trigram")(embedded_sequences)
    cnn4=Conv1D(filters=50, kernel_size=4, strides=1, padding="same", activation="tanh", name="CNN_4gram")(embedded_sequences)

    # max pooling over all words in the document
    maxpool2=GlobalMaxPooling1D()(cnn2)
    maxpool3=GlobalMaxPooling1D()(cnn3)
    maxpool4=GlobalMaxPooling1D()(cnn4)

    x=Concatenate()([maxpool2, maxpool3, maxpool4])

    x=Dropout(0.2)(x)
    x=Dense(50)(x)
    x=Dense(label_num, activation="softmax")(x)

    model = Model(inputs=word_sequence_input, outputs=x)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def baseline_reference_cnn_model(vocab_size, word_embedding_dim, label_num):
    word_sequence_input = Input(shape=(None,), dtype='int32')
    word_embedding_layer = Embedding(vocab_size,
                                    word_embedding_dim,
                                    trainable=True)
    embedded_sequences = word_embedding_layer(word_sequence_input)

    cnn2=Conv1D(filters=50, kernel_size=2, strides=1, padding="same", activation="tanh", name="CNN_bigram")(embedded_sequences)
    cnn3=Conv1D(filters=50, kernel_size=3, strides=1, padding="same", activation="tanh", name="CNN_trigram")(embedded_sequences)
    cnn4=Conv1D(filters=50, kernel_size=4, strides=1, padding="same", activation="tanh", name="CNN_4gram")(embedded_sequences)

    # max pooling over all words in the document
    maxpool2=GlobalMaxPooling1D()(cnn2)
    maxpool3=GlobalMaxPooling1D()(cnn3)
    maxpool4=GlobalMaxPooling1D()(cnn4)

    x=Concatenate()([maxpool2, maxpool3, maxpool4])

    x=Dropout(0.2)(x)
    x=Dense(50)(x)
    x=Dense(label_num, activation="softmax")(x)

    model = Model(inputs=word_sequence_input, outputs=x)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

if __name__ == '__main__':
    pass