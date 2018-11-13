from keras.models import Model
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Input, TimeDistributed
from keras_contrib.utils import save_load_utils
from attention_keras import Position_Embedding, Attention
from keras_contrib.layers.crf import CRF

def build_model(input_length, emb_input_dim, emb_out_dim, lstm_hidden_units, num_cls, embedding_matrix=None):

    l_input = Input(shape=(input_length,))

    if embedding_matrix is None:
        l_emb = Embedding(emb_input_dim, emb_out_dim)(l_input)
        #l_emb = Embedding(emb_input_dim, emb_out_dim, mask_zero=True)(l_input)
    else:
        l_emb = Embedding(emb_input_dim, emb_out_dim, weights=[embedding_matrix], trainable=True)(l_input)

    # add bilstm layer
    l_posemb = Position_Embedding()(l_emb)
    l_posemb = Dropout(0.1)(l_posemb)

    l_bilstm = Bidirectional(LSTM(lstm_hidden_units, return_sequences=True))(l_posemb)

    # add attention layer
    l_att = Attention(nb_head=8, size_per_head=32)([l_bilstm, l_bilstm, l_bilstm])
    print('l_att.shape:', l_att.shape)

    # add dense layer
    l_dense = TimeDistributed(Dense(num_cls))(l_att)

    crf = CRF(num_cls)
    l_crf = crf(l_dense)
    model = Model(l_input, l_crf)

    model.compile('rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
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


