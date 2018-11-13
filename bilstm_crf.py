from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils

DROPOUT_RATE = 0.1

def build_model(input_length, emb_input_dim, emb_out_dim, lstm_hidden_units, num_cls, embedding_matrix=None):
    model = Sequential()

    if embedding_matrix is None:
        model.add(Embedding(emb_input_dim, emb_out_dim, mask_zero=True))
    else:
        model.add(Embedding(emb_input_dim, emb_out_dim, weights=[embedding_matrix], trainable=True))

    model.add(Bidirectional(LSTM(lstm_hidden_units, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))

    model.add(TimeDistributed(Dense(num_cls)))

    crf_layer = CRF(num_cls)
    model.add(crf_layer)

    model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    model.summary()

    return model

def save_model(model, filename):
    save_load_utils.save_all_weights(model, filename)

def load_model(filename):
    model = build_model()
    save_load_utils.load_all_weights(model, filename)
    return model

def train():
    build_model(80, 100, 100, 100, 31)

def main():
    train()

if __name__ == '__main__':
    main()


