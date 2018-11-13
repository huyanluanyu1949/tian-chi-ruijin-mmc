from keras.models import Model
from keras.layers import Embedding, Dense, Dropout, Input, SeparableConv1D, Conv1D
from keras_contrib.utils import save_load_utils
from attention_keras import Position_Embedding, Attention

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

    l_cnn = SeparableConv1D(128, 3, use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(l_posemb)
    l_cnn = SeparableConv1D(128, 3, use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(l_cnn)
    l_cnn = Conv1D(128, 3, use_bias=False, padding='same', kernel_initializer='he_normal')(l_cnn)

    # add attention layer
    l_att = Attention(nb_head=64, size_per_head=16)([l_cnn] * 3)
    print('l_att.shape:', l_att.shape)

    # add dense layer
    l_dense = Dense(num_cls, activation='softmax')(l_att)

    model = Model(l_input, l_dense)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
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


